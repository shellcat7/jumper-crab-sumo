"""Environment config for jumper.gyoji_walk (jumper-challenge-2026).

The gyoji's legs. In the ring the gyoji (`sumo/gyoji.py`) carries the gunbai in its
right claw -- the right-front arm held up in front (`arms.GYOJI_HOLD`), raised to
call the winner (`arms.GYOJI_RAISE`) -- and walks on the other five legs with the
official posture policy, which was trained to walk on six. Every step that policy
plans for the right-front foot finds no foot, and the body drops onto the missing
corner: on the 10/13-14 reels the gyoji's base bobbed 5.3-5.5 mm (its height less a
0.5 s moving average), the wrestlers' 1.7-2.6 mm.

This is `jumper.posture` with the right-front arm taken away from the policy and
held where the gunbai holds it, and the gait asked for changed to one five legs can
walk -- `jumper.five_foot`'s, mirrored (that task carries the *left* arm):

    1. the arm is held   `GunbaiJointPositionAction`: the policy still sends 20
                         targets (so a checkpoint swaps in for the gyoji's walker
                         unchanged), and the right-front arm's four are overwritten
                         with the gunbai pose -- held in most episodes, raised in
                         `RAISE_SHARE` of them, so the legs know both
    2. the gait          the tripod (LF+RM+LR against RF+LM+RR) cannot be walked
                         without RF; `gunbai_gait` alternates RM+LR against LM+RR
                         with the free front claw (LF) planted throughout, as
                         five_foot plants its free front claw
    3. the tripod's      `stance_load` and `stance_ground_gap` are written for the
       stance terms      tripod groups: off; the gait term's support half carries
                         what they asked
    4. the bob           a cost on the base's vertical speed and its roll and pitch
                         rates, the thing to be fixed

Everything else -- the commands, tracking, posture, energy, foot terms, the clock in
the observation -- is posture's. A foot term for the right-front claw costs nothing
or a constant (it never touches down), and gives no gradient either way.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..common.mdp.rewards import contact_state, cycling_gate, moving_gate, phase_match
from ..posture.env_cfg import env_cfg as posture_env_cfg
from ..tripod.mdp.phase import GAIT_FREQ_HZ, gait_phase

#: Episodes in which the gunbai is raised (the call) rather than held.
RAISE_SHARE = 0.2
#: The two swing groups and the planted claw, in the contact sensor's leg order
#: (LF, RF, LM, RM, LR, RR) with RF -- the gunbai -- taken out: LF 0, LM 1, RM 2,
#: LR 3, RR 4.
FIVE = (0, 2, 3, 4, 5)
SWING_A = (2, 3)        # RM, LR
SWING_B = (1, 4)        # LM, RR
#: The bob's cost, per second: (vertical speed m/s)^2 + 0.1 (roll and pitch rate rad/s)^2.
BOB_WEIGHT = -20.0


@dataclass(kw_only=True)
class GunbaiJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env) -> "GunbaiJointPositionAction":
        return GunbaiJointPositionAction(self, env)


class GunbaiJointPositionAction(JointPositionAction):
    """Posture's joint action, the right-front arm overwritten with the gunbai pose."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raised = torch.zeros(self.num_envs, device=self.device)

    def process_actions(self, actions: torch.Tensor) -> None:
        from ..sumo.arms import GYOJI_HOLD, GYOJI_RAISE, overlay

        super().process_actions(actions)
        overlay(self, GYOJI_HOLD, torch.ones_like(self._raised))
        overlay(self, GYOJI_RAISE, self._raised)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        n = self._raised[ids].shape[0]
        self._raised[ids] = (torch.rand(n, device=self.device) < RAISE_SHARE).float()


def gunbai_phase_score(contact: torch.Tensor, phase: torch.Tensor, sigma: float = 0.7) -> torch.Tensor:
    """`tripod_phase_score` for five legs: the clock's first half swings RM+LR, the
    second LM+RR; the other legs in support, LF always. RF is not scored."""
    five = contact[:, list(FIVE)]
    return torch.where(phase < 0.5, phase_match(five, SWING_A, sigma), phase_match(five, SWING_B, sigma))


def gunbai_gait(env, sensor_name: str, command_name: str, command_threshold: float = 0.05,
                sigma: float = 0.7, max_air_time: float = 0.5, max_contact_time: float = 1.0,
                freq_hz: float = GAIT_FREQ_HZ) -> torch.Tensor:
    """`tripod_gait` with `gunbai_phase_score`: the same clock, gates and sigma."""
    contact = contact_state(env, sensor_name)
    phase = gait_phase(env, freq_hz)
    return (gunbai_phase_score(contact, phase, sigma)
            * moving_gate(env, command_name, command_threshold)
            * cycling_gate(env, sensor_name, max_air_time, max_contact_time))


def body_bob(env) -> torch.Tensor:
    d = env.scene["robot"].data
    return d.root_link_lin_vel_w[:, 2].square() + 0.1 * d.root_link_ang_vel_b[:, :2].square().sum(-1)


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = posture_env_cfg(asset=asset, play=play)
    base = cfg.actions["joint_pos"]
    import dataclasses

    cfg.actions["joint_pos"] = GunbaiJointPositionActionCfg(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init})
    gait = cfg.rewards.pop("tripod_gait")
    cfg.rewards["gunbai_gait"] = RewardTermCfg(func=gunbai_gait, weight=gait.weight, params=dict(gait.params))
    for name in ("stance_load", "stance_ground_gap"):
        cfg.rewards.pop(name, None)
    weight = float(os.environ.get("JUMPER_GYOJI_BOB", BOB_WEIGHT))
    cfg.rewards["body_bob"] = RewardTermCfg(func=body_bob, weight=weight)
    return cfg
