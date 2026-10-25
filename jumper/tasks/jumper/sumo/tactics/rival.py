"""A frozen tactician as the opponent (jumper-challenge-2026).

`opponent.OpponentPolicyAction` drives the opponent with a walking policy told to
walk at the robot. This drives it the way the robot is driven in
`jumper.sumo.tactics`: a frozen tactician chooses six numbers from what it sees
of the bout -- from the opponent's side now -- and they become commands for a
frozen walking policy. So any trained tactician can stand on the west side:
tactician against tactician, for a round robin or for training against past
versions of oneself.

    JUMPER_SUMO_OPP_TACTICIAN      the tactician checkpoint for the west side
    JUMPER_SUMO_OPP_TACTICIAN_LOW  its walking policy (default: the official one)
    JUMPER_SUMO_OPP_TREND=1        the tactician was trained with the opponent
                                   trend input (JUMPER_TACTICS_TREND=1)

Works for `jumper.sumo` (the robot is a walker) and `jumper.sumo.tactics` (the
robot is a tactician) alike; `env_cfg.py` of `jumper.sumo` calls
`use_tactician_opponent` when the first variable is set.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from ..opponent import OPPONENT, FrozenActor, OpponentPolicyActionCfg
from .mdp import (
    ACTION_DIM, TacticsAction, bout_phase, opponent_relative, opponent_trend, own_motion,
    ring_state,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def tactician_choice(env: "ManagerBasedRlEnv", action_name: str = OPPONENT) -> torch.Tensor:
    """The frozen tactician's last raw output -- what `last_action` is to the
    robot's tactician, which saw the action manager's (unclipped) action."""
    return env.action_manager.get_term(action_name)._raw


def tactician_group(radius: float, me: str = OPPONENT, other: str = "robot",
                    trend: bool = False, claws: bool = False, memory: bool = False) -> ObservationGroupCfg:
    """The tactician's observation, seen from `me`; the terms in the order
    `tactics/env_cfg.py` builds them, so a checkpoint's inputs line up."""
    terms = {
        "own_motion": ObservationTermCfg(func=own_motion, params={"me": me}),
        "opponent": ObservationTermCfg(func=opponent_relative, params={"me": me, "other": other}),
        "ring": ObservationTermCfg(func=ring_state, params={"radius": radius, "me": me, "other": other}),
        "bout": ObservationTermCfg(func=bout_phase),
        "actions": ObservationTermCfg(func=tactician_choice, params={"action_name": me}),
    }
    if trend:
        terms["trend"] = ObservationTermCfg(func=opponent_trend, params={"me": me, "other": other})
    if claws:
        from .waza import claws_state

        terms["claws"] = ObservationTermCfg(func=claws_state, params={"me": me})
    if memory:
        from .mdp import bout_memory

        terms["memory"] = ObservationTermCfg(func=bout_memory, params={"me": me, "other": other})
    return ObservationGroupCfg(terms=terms)


@dataclass(kw_only=True)
class OpponentTacticsActionCfg(OpponentPolicyActionCfg):
    """`checkpoint` is the walking policy; `tactician` the frozen tactician."""

    tactician: str = ""
    tactician_group: object = None

    def build(self, env: "ManagerBasedRlEnv") -> "OpponentTacticsAction":
        return OpponentTacticsAction(self, env)


class OpponentTacticsAction(TacticsAction):
    cfg: OpponentTacticsActionCfg

    def __init__(self, cfg: OpponentTacticsActionCfg, env: "ManagerBasedRlEnv"):
        self._tactician = FrozenActor(Path(cfg.tactician), str(env.device))
        #: As many outputs as the checkpoint has: 6, or 13 with claws (waza.py).
        self._dim = self._tactician.act_dim
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, self._dim, device=self.device)
        self._tac_obs = None

    @property
    def action_dim(self) -> int:
        # Nothing for the learner to send: hidden like the walking opponent.
        return 0 if getattr(self, "_hidden", False) else self._action_dim

    def _tactician_obs(self) -> torch.Tensor:
        if self._tac_obs is None:
            from mjlab.managers.observation_manager import ObservationManager

            self._tac_obs = ObservationManager({"tactician": self.cfg.tactician_group}, self._env)
        obs = self._tac_obs.compute_group("tactician", update_history=True)
        assert obs.shape[-1] == self._tactician.obs_dim, (
            f"rival tactician observation is {obs.shape[-1]} wide, its policy takes "
            f"{self._tactician.obs_dim} (JUMPER_SUMO_OPP_TREND?)")
        return obs

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions  # the learner sends none
        self._raw = self._tactician(self._tactician_obs())
        super().process_actions(self._raw)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        if self._tac_obs is not None:
            self._tac_obs.reset(env_ids)


def use_tactician_opponent(cfg, radius: float) -> None:
    """Swap the opponent's walking-policy action for a frozen tactician."""
    base = cfg.actions[OPPONENT]
    kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(OpponentPolicyActionCfg)
              if f.init}
    official = Path(__file__).resolve().parents[2] / "posture/out/example/model_74800.pt"
    kwargs["checkpoint"] = os.environ.get("JUMPER_SUMO_OPP_TACTICIAN_LOW") or str(official)
    kwargs["approach_speed_range"] = None
    cfg.actions[OPPONENT] = OpponentTacticsActionCfg(
        **kwargs,
        tactician=os.environ["JUMPER_SUMO_OPP_TACTICIAN"],
        tactician_group=tactician_group(
            radius, trend=os.environ.get("JUMPER_SUMO_OPP_TREND", "") == "1",
            claws=os.environ.get("JUMPER_SUMO_OPP_WAZA", "") == "1",
            memory=os.environ.get("JUMPER_SUMO_OPP_MEMORY", "") == "1"),
    )
