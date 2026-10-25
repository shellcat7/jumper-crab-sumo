"""A league opponent: every bout, one of a whole roster of wrestlers (jumper-challenge-2026).

`opponent.py`'s pool draws among walking policies; `rival.py` puts one frozen
tactician on the west side. This draws, every bout and per environment, among a
roster that mixes both -- walkers that walk at the robot, and tacticians that
steer their own walking policy -- so one learner can be trained against everything
trained so far: the "comprehensive" wrestler of 10/10 evening, against the
pusher, the reader, the puller and the circler in turn.

    JUMPER_SUMO_LEAGUE   a roster file, the round robin's format
                         (name|kind|checkpoint|walking policy|trend|waza|memory)

A wrestler with a memory (the 7th field 1, a claw wrestler trained with
JUMPER_TACTICS_MEMORY=1, 10/15 night) reads the bout's last second as well, so the
learner can meet the strongest so far -- itself frozen -- in the league.

Per step: every tactician in the roster is run on the tactician observation
(seen from the opponent; with the trend input, and its first 28 numbers for a
tactician trained without), each environment keeps its own entry's choice; the
commands are a tactician's mapped choice or a walker's walk-at-the-robot; then
every walking policy in the roster runs on the opponent's posture observation and
each environment keeps its own entry's. All on the GPU, a handful of small MLPs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction

from ..mdp import SHIKIRI_POSTURE, bout, hold_mark, shikiri_or
from ..opponent import OPPONENT, FrozenActor, OpponentPolicyActionCfg, approach_command
from .mdp import ACTION_DIM, TacticsAction
from .rival import OpponentTacticsActionCfg, tactician_group

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: Width of the tactician observation without the trend input.
NO_TREND_DIM = 28


@dataclass(kw_only=True)
class LeagueOpponentActionCfg(OpponentTacticsActionCfg):
    """`checkpoint`: the roster's walking policies (comma-separated, each once);
    `entries`: (walking-policy index, tactician path or "") per wrestler."""

    entries: list = field(default_factory=list)
    names: list = field(default_factory=list)
    #: Whether any wrestler in the roster reads a memory (its 7th field 1).
    memory: bool = False
    #: Per entry, the claws' lean for a claw wrestler (the roster's waza field: 1, 1:x,
    #: 2 for 0), or None for one without claws (10/14 night: the league can field the
    #: claw wrestlers -- 鋏ノ富士, 正面山, ... -- and the learner meets them).
    leans: list = field(default_factory=list)
    #: The entity a walker in the roster walks at: the robot, for the west side;
    #: the opponent, for a league on the east side too (`use_league`, the gyoji's
    #: training, where neither wrestler learns).
    target: str = "robot"

    def build(self, env: "ManagerBasedRlEnv") -> "LeagueOpponentAction":
        return LeagueOpponentAction(self, env)


class LeagueOpponentAction(TacticsAction):
    cfg: LeagueOpponentActionCfg

    def __init__(self, cfg: LeagueOpponentActionCfg, env: "ManagerBasedRlEnv"):
        self._entries = list(cfg.entries)
        tac_paths = sorted({t for _, t in self._entries if t})
        self._tacticians = [FrozenActor(Path(p), str(env.device)) for p in tac_paths]
        wide = [t for t in self._tacticians if t.obs_dim > CLAW_TREND_DIM]
        if wide and not cfg.memory:
            raise ValueError("a tactician in the league reads a memory: mark it in the roster's 7th field")
        self._tac_index = {p: i for i, p in enumerate(tac_paths)}
        n = env.num_envs
        self._entry = torch.zeros(n, dtype=torch.long, device=env.device)
        self._tac = torch.full((n,), -1, dtype=torch.long, device=env.device)
        leans = list(cfg.leans) or [None] * len(self._entries)
        self._claws = any(x is not None for x in leans)
        #: 13 outputs when the roster has claw wrestlers: theirs; the others' last 7 are 0.
        self._dim = ACTION_DIM + (7 if self._claws else 0)
        self._entry_lean = torch.tensor([x if x is not None else 0.0 for x in leans], device=env.device)
        self._entry_claws = torch.tensor([x is not None for x in leans], device=env.device)
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, self._dim, device=self.device)
        self._tac_obs = None
        #: Bouts so far against each roster entry (for logs).
        self.drawn = torch.zeros(len(self._entries), dtype=torch.long, device=self.device)

    @property
    def action_dim(self) -> int:
        return 0 if getattr(self, "_hidden", False) else self._action_dim

    def _draw(self, env_ids: torch.Tensor) -> None:
        super()._draw(env_ids)
        if not getattr(self, "_entries", None):
            return
        e = torch.randint(len(self._entries), (len(env_ids),), device=self.device)
        self._entry[env_ids] = e
        walker = torch.tensor([w for w, _ in self._entries], device=self.device)
        tac = torch.tensor([self._tac_index[t] if t else -1 for _, t in self._entries],
                           device=self.device)
        self._which[env_ids] = walker[e]
        self._tac[env_ids] = tac[e]
        if hasattr(self, "drawn"):
            self.drawn += torch.bincount(e, minlength=len(self._entries))

    def _tactician_obs(self) -> torch.Tensor:
        if self._tac_obs is None:
            from mjlab.managers.observation_manager import ObservationManager

            self._tac_obs = ObservationManager({"tactician": self.cfg.tactician_group}, self._env)
        return self._tac_obs.compute_group("tactician", update_history=True)

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions
        env = self._env
        state = bout(env)
        state.update()
        # 1. Tacticians: each on its own envs.
        raw = torch.zeros(self.num_envs, self._dim, device=self.device)
        if self._tacticians:
            obs = self._tactician_obs()
            for k, tac in enumerate(self._tacticians):
                mask = self._tac == k
                if bool(mask.any()):
                    raw[mask, :tac.act_dim] = tac(_inputs_for(tac.obs_dim, obs)[mask])
        self._raw = raw
        self._choice = raw.clamp(-1.0, 1.0)
        if self._claws:
            # The claw wrestlers' seven claw outputs, with their own lean; the others'
            # claws idle (0, no lean).
            from .waza import SIDES, waza

            w = waza(env)
            if w is not None:
                has = self._entry_claws[self._entry]
                side = SIDES.index(self.cfg.entity_name)
                w.lean_env[side] = torch.where(has, self._entry_lean[self._entry], torch.zeros_like(self._entry_lean[self._entry]))
                w.command(side, torch.where(has[:, None], self._choice[:, ACTION_DIM:], torch.zeros_like(self._choice[:, ACTION_DIM:])))
        # 2. Commands: a tactician's mapped choice, or a walker's walk at the robot.
        tac_twist, tac_posture = TacticsAction.commands(self)
        walk = approach_command(self._entity, env.scene[self.cfg.target], self._speed)
        neutral = torch.tensor([0.0, 0.0, 0.0, self.cfg.neutral_height], device=self.device)
        crouch = neutral + torch.tensor(SHIKIRI_POSTURE, device=self.device)
        walk_twist = shikiri_or(walk, state.started, hold_mark(env, self.cfg.entity_name))
        walk_posture = shikiri_or(neutral.expand(self.num_envs, 4), state.started, crouch)
        is_tac = (self._tac >= 0)[:, None]
        env.command_manager.get_term(self.cfg.twist_name).set(torch.where(is_tac, tac_twist, walk_twist))
        env.command_manager.get_term(self.cfg.posture_name).set(
            torch.where(is_tac, tac_posture, walk_posture))
        # 3. Walking policies: each env its own entry's.
        JointPositionAction.process_actions(self, self._act(self._observe()))
        self._overlay_arms()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0.0
        if self._tac_obs is not None:
            self._tac_obs.reset(env_ids)


#: The tactician observation with the trend and the claws (60): own motion 6, opponent 8,
#: ring 6, bout 2, actions 13, trend 3, claws 22. A tactician without claws reads the
#: first 22, the first 6 of the actions, and (trained with it) the trend. A league with
#: a memory wrestler appends the memory (96) after these 60.
CLAW_TREND_DIM = 60
_NO_CLAW_TREND = list(range(28)) + [35, 36, 37]


def _inputs_for(width: int, obs: torch.Tensor) -> torch.Tensor:
    """The columns of the league's tactician observation a tactician of `width`
    inputs was trained on."""
    if width == obs.shape[-1]:
        return obs
    if obs.shape[-1] > CLAW_TREND_DIM and width == CLAW_TREND_DIM:
        return obs[:, :CLAW_TREND_DIM]
    if obs.shape[-1] >= CLAW_TREND_DIM and width == 31:
        return obs[:, _NO_CLAW_TREND]
    return obs[:, :NO_TREND_DIM]


def _lean(field: str):
    """The roster's waza field: '' / '0' no claws (None), '1' lean 1, '2' lean 0, '1:x'."""
    field = (field or "").strip()
    if field in ("", "0"):
        return None
    if field == "2":
        return 0.0
    if ":" in field:
        return float(field.split(":", 1)[1])
    return 1.0


def _native(path: str) -> str:
    """A Git Bash path (/c/Users/...) as Windows Python opens it; others unchanged."""
    import os
    import re

    m = re.match(r"^/([a-zA-Z])/(.*)$", path)
    return f"{m.group(1).upper()}:/{m.group(2)}" if (m and os.name == "nt") else path


def use_league_opponent(cfg, radius: float, roster: str) -> None:
    """Swap the opponent's action for a roster drawn every bout (module docstring)."""
    use_league(cfg, radius, roster, OPPONENT)


def use_league(cfg, radius: float, roster: str, side: str, speed_range=None) -> None:
    """A roster drawn every bout for `side` ("opponent", or "robot" when no one on
    the east side learns): the side's action term, named after it, is replaced.
    For the robot the term is new (the learner's `joint_pos` is the caller's to
    remove), and its walking policies get the robot's own posture observation."""
    from ..opponent import opponent_observation_group

    official = str(Path(__file__).resolve().parents[2] / "posture/out/example/model_74800.pt")
    lines = [l.strip() for l in Path(roster).read_text(encoding="utf-8").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    walkers: list[str] = []
    entries, names, leans = [], [], []
    memory = False
    for line in lines:
        name, kind, ckpt, walking, _trend, waza_field, mem = (line.split("|") + [""] * 7)[:7]
        leans.append(_lean(waza_field) if kind == "tactician" else None)
        if kind == "tactician" and mem.strip() == "1":
            if leans[-1] is None:
                raise ValueError(f"{name}: a memory wrestler in the league must be a claw wrestler")
            memory = True
        ckpt, walking = _native(ckpt), _native(walking) if walking else walking
        walk = ckpt if kind == "walker" else (walking or official)
        if walk not in walkers:
            walkers.append(walk)
        entries.append((walkers.index(walk), ckpt if kind == "tactician" else ""))
        names.append(name)
    base = cfg.actions[OPPONENT]
    kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(OpponentPolicyActionCfg)
              if f.init}
    kwargs["checkpoint"] = ",".join(walkers)
    other = "robot" if side == OPPONENT else OPPONENT
    if side != OPPONENT:
        kwargs.update(entity_name=side, twist_name="twist", posture_name="posture",
                      observation_group=opponent_observation_group(
                          cfg.observations["actor"], entity=side, twist="twist",
                          posture="posture", action=side))
    if speed_range is not None:
        kwargs["approach_speed_range"] = speed_range
    cfg.actions[side] = LeagueOpponentActionCfg(
        **kwargs, tactician="",
        tactician_group=tactician_group(radius, me=side, other=other, trend=True,
                                        claws=any(x is not None for x in leans), memory=memory),
        entries=entries, names=names, leans=leans, target=other, memory=memory)
