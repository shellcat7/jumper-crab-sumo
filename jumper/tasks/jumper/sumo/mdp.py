"""MDP terms of jumper.sumo (jumper-challenge-2026).

The learner keeps the posture robot's observation and actions, so what makes it a
sumo wrestler is what it is asked and what it is paid:

- `ApproachCommand` replaces the random velocity command: every step it is "face
  the opponent and walk at it" at a speed drawn per episode. The posture task's
  tracking rewards then pay for making ground towards the opponent -- which, once
  the two are in contact, means pushing it back.
- `bout_outcome` pays at the step a bout ends: + when the opponent leaves the ring
  or falls, - when the learner does (weights in `env_cfg.py`).
- `push_out` pays while the opponent moves away from the ring's centre.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .opponent import OPPONENT, approach_command

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


#: Both robots' foot contact sensors; the first two feet of each are the claws'
#: palm pads (`common/constants.py`, FOOT_GEOMS order: LF, RF, LM, RM, LR, RR).
FEET_SENSORS = ("feet_ground_contact", "opponent_feet_ground_contact")
#: Shikiri posture: body lowered and pitched nose-down, claws pressed to the clay.
#: `[twist, pitch, roll, height]`; height relative to standing.
SHIKIRI_POSTURE = (0.0, 0.10, 0.0, -0.012)


class BoutState:
    """Shikiri and tachiai, one per environment (kept on the env as `sumo_bout`).

    A bout starts crouched and still, each wrestler with its claws raised in front
    of it (`arms.py`). Each lowers them at its own moment, `lower_at` (drawn per
    bout and per wrestler, so neither the start nor the other can be timed from
    the clock), over `arms.LOWER_S`. The bout begins -- the tachiai -- once both
    have finished lowering and both robots have had both claws on the ground for
    `hands_down_s` without a break; after `max_shikiri_s` it begins regardless, so
    a robot that lifts a claw cannot stall the bout (with a gyoji, the gyoji calls
    matta instead, `referee.py`). Until then both are commanded to stand in the
    shikiri posture; afterwards to walk at each other.
    """

    def __init__(self, env: "ManagerBasedRlEnv", hands_down_s: float = 0.02,
                 lower_at: tuple[float, float] = (0.6, 1.4), max_shikiri_s: float = 4.0):
        n, dev = env.num_envs, env.device
        self.env = env
        from .arms import ceremony_s

        #: Replays (`JUMPER_SUMO_CEREMONY`, 10/20): seconds of the claws' ceremony
        #: (`arms.ceremony_offsets`) before the shikiri proper; everything after it shifts.
        self.ceremony = ceremony_s()
        lower_at = (lower_at[0] + self.ceremony, lower_at[1] + self.ceremony)
        max_shikiri_s += self.ceremony
        self.hands_down_s, self.lower_range, self.max_shikiri_s = hands_down_s, lower_at, max_shikiri_s
        self.started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.elapsed = torch.zeros(n, device=dev)
        self.hands = torch.zeros(n, device=dev)
        #: [N, 2]: when the robot (0) and the opponent (1) begin lowering their claws.
        self.lower_at = torch.empty(n, 2, device=dev).uniform_(*lower_at)
        self.wait = self._wait()
        #: With a gyoji (`gyoji.py` sets it): the shikiri lines' inner edge, m from
        #: the centre. Claws down in front of the line are not hands down.
        self.line_x: float | None = None
        #: [N, 2, 2]: each wrestler's mark -- where it was put at the reset, behind
        #: its line -- which it is held to through the shikiri (`hold_mark`).
        self.mark = torch.zeros(n, 2, 2, device=dev)
        self.marked = torch.zeros(n, dtype=torch.bool, device=dev)
        #: The step each bout began at (-1 until it has), for overlays and logs.
        self.start_step = torch.full((n,), -1, dtype=torch.long, device=dev)
        self._step: int | None = None

    def _wait(self) -> torch.Tensor:
        """Shikiri time until both wrestlers have their claws down."""
        from .arms import LOWER_S

        return self.lower_at.max(-1).values + LOWER_S

    def update(self) -> None:
        """Advance once per environment step, whoever asks first."""
        env = self.env
        step = int(env.common_step_counter)
        if step == self._step:
            return
        self._step = step
        from ..common.mdp.rewards import contact_state

        fresh = ~self.marked
        if bool(fresh.any()):
            origin = env.scene.env_origins[:, :2]
            for col, entity in enumerate(("robot", OPPONENT)):
                pos = env.scene[entity].data.root_link_pos_w[:, :2] - origin
                self.mark[fresh, col] = pos[fresh]
            self.marked |= fresh
        down = torch.ones_like(self.started)
        for sensor in FEET_SENSORS:
            down &= contact_state(env, sensor)[:, :2].bool().all(dim=-1)
        if self.line_x is not None:
            down &= claws_behind_line(env, self.line_x)
        # Hands count once both have begun lowering (a raised claw that brushes the
        # clay is not hands down), and the bout begins the moment both pairs are
        # down -- as in a real bout, the tachiai is the second wrestler's hands
        # touching. `hands_down_s` is only a debounce (4 steps). Until 10/10 evening
        # the hands had to stay down 0.3 s after both had finished lowering.
        down &= self.elapsed >= self.lower_at.max(-1).values
        waiting = ~self.started
        self.elapsed[waiting] += env.step_dt
        self.hands = torch.where(down & waiting, self.hands + env.step_dt, torch.zeros_like(self.hands))
        go = waiting & ((self.hands >= self.hands_down_s) | (self.elapsed >= self.max_shikiri_s))
        self.started |= go
        self.start_step[go] = step

    def reset(self, env_ids: torch.Tensor) -> None:
        self.started[env_ids] = False
        self.elapsed[env_ids] = 0.0
        self.hands[env_ids] = 0.0
        self.start_step[env_ids] = -1
        self.lower_at[env_ids] = torch.empty(len(env_ids), 2, device=self.env.device).uniform_(*self.lower_range)
        self.marked[env_ids] = False
        self.wait = self._wait()


CLAW_PADS = ("LF_palm_pad_b_link", "RF_palm_pad_b_link")


def claws_behind_line(env: "ManagerBasedRlEnv", line_x: float) -> torch.Tensor:
    """Both wrestlers' claw pads on or behind their own shikiri line: the robot's
    at x <= -line_x, the opponent's at x >= +line_x (ring frame)."""
    cache = env.__dict__.setdefault("_sumo_claw_ids", {})
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for entity, sign in (("robot", -1.0), (OPPONENT, 1.0)):
        ent = env.scene[entity]
        if entity not in cache:
            names = list(ent.body_names)
            cache[entity] = torch.tensor([names.index(n) for n in CLAW_PADS], device=env.device)
        x = ent.data.body_link_pos_w[:, cache[entity], 0] - env.scene.env_origins[:, None, 0]
        ok &= (sign * x >= line_x).all(-1)
    return ok


def bout(env: "ManagerBasedRlEnv") -> BoutState:
    state = getattr(env, "sumo_bout", None)
    if state is None:
        state = BoutState(env)
        env.sumo_bout = state
    return state


def reset_bout(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> None:
    """Event term (mode `reset`): a new bout starts in shikiri."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    bout(env).reset(env_ids)


#: Holding a wrestler to its mark during the shikiri: speed per metre off it, the
#: most it is asked to walk, and how close counts as on it.
HOLD_GAIN, HOLD_MAX, HOLD_DEADBAND = 2.0, 0.12, 0.01


def hold_mark(env: "ManagerBasedRlEnv", entity: str) -> torch.Tensor:
    """`[vx, vy, wz]` (body frame) for the shikiri: back to one's own mark, facing
    the other's. A still command was not enough: a policy trained to push creeps
    forward while it waits (sumo_cont700, 10/10: from 0.36 m to 0.21 m, its claws
    far over the line, so the gyoji never saw its hands down and called matta
    three times). In a real bout the wrestlers go back to their lines after one."""
    state = bout(env)
    col = 0 if entity == "robot" else 1
    me = env.scene[entity].data
    pos = me.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    err = state.mark[:, col] - pos
    dist = torch.linalg.norm(err, dim=-1, keepdim=True)
    v = HOLD_GAIN * err * (HOLD_MAX / (HOLD_GAIN * dist).clamp(min=HOLD_MAX)) * (dist > HOLD_DEADBAND)
    w, x, y, z = me.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    face = state.mark[:, 1 - col] - pos
    err_yaw = torch.remainder(torch.atan2(face[:, 1], face[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1],
                        torch.clamp(2.0 * err_yaw, -0.5, 0.5)), -1)


def shikiri_or(command: torch.Tensor, started: torch.Tensor, still) -> torch.Tensor:
    """`command` where the bout has started, `still` (broadcast) where not."""
    still = torch.as_tensor(still, device=command.device, dtype=command.dtype).expand_as(command)
    return torch.where(started.unsqueeze(-1), command, still)


class ApproachCommand(CommandTerm):
    """`[vx, vy, wz]` that turns the robot to face the opponent and walks at it.

    Zero during shikiri (see `BoutState`), when it also holds the posture command
    named by `posture_name` at `SHIKIRI_POSTURE`, and neutral after the tachiai.
    """

    cfg: "ApproachCommandCfg"

    def __init__(self, cfg: "ApproachCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._cmd = torch.zeros(self.num_envs, 3, device=self.device)
        self._speed = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._cmd

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        lo, hi = self.cfg.speed_range
        self._speed[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi)
        self._update_command(env_ids)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        env = self._env
        state = bout(env)
        state.update()
        me = env.scene[self.cfg.entity_name]
        other = env.scene[self.cfg.target_name]
        cmd = approach_command(me, other, 1.0, self.cfg.turn_gain, self.cfg.max_turn)
        cmd[:, 0] *= self._speed
        cmd = shikiri_or(cmd, state.started, hold_mark(self._env, "robot"))
        if env_ids is None:
            self._cmd[:] = cmd
        else:
            self._cmd[env_ids] = cmd[env_ids]
        if self.cfg.posture_name:
            neutral = torch.tensor([0.0, 0.0, 0.0, self.cfg.neutral_height], device=self.device)
            crouch = neutral + torch.tensor(SHIKIRI_POSTURE, device=self.device)
            posture = shikiri_or(neutral.expand(self.num_envs, 4), state.started, crouch)
            env.command_manager.get_term(self.cfg.posture_name).set(posture)


@dataclass(kw_only=True)
class ApproachCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    target_name: str = OPPONENT
    speed_range: tuple[float, float] = (0.2, 0.45)
    turn_gain: float = 2.0
    max_turn: float = 0.75
    resampling_time_range: tuple[float, float] = (4.0, 8.0)
    #: The posture command (a `HeldCommand`) this term drives through shikiri;
    #: empty to leave posture alone (replay, where an operator owns it).
    posture_name: str = "posture"
    neutral_height: float = 0.10647

    def build(self, env: "ManagerBasedRlEnv") -> ApproachCommand:
        return ApproachCommand(self, env)


# ── Losing: a foot outside the straw, or anything but a foot on the clay ──────
#
# The rules of a real bout, read for a six-legged wrestler: the bout is lost the
# moment any part of the body touches the ground outside the ring -- a toe is
# enough, and stepping *on* the straw is still inside -- or any part other than
# the soles touches the ground anywhere (`touched_down`). Jumper's soles are its
# six feet -- the two claw palm pads it walks on and the four foot tips -- for
# stepping out; for going down, the rest of the claws and the calves count as
# soles too, since they touch the clay in ordinary walking (env_cfg's
# `*_body_ground` sensors): down is the body or an upper leg segment.

#: Each wrestler's foot ground-contact sensor and its everything-else one.
FEET_SENSOR = {"robot": "feet_ground_contact", OPPONENT: "opponent_feet_ground_contact"}
BODY_SENSOR = {"robot": "robot_body_ground", OPPONENT: "opponent_body_ground"}


def _feet_xy(env: "ManagerBasedRlEnv", entity: str) -> torch.Tensor:
    """[N, feet, 2] foot positions relative to the ring's centre, in the column
    order of the entity's foot contact sensor (its `primary_names`)."""
    cache = env.__dict__.setdefault("_sumo_feet_ids", {})
    ent = env.scene[entity]
    if entity not in cache:
        names = list(ent.body_names)
        sensor = env.scene.sensors[FEET_SENSOR[entity]]
        links = [n.split("/")[-1].removesuffix("_meshcol") for n in sensor.primary_names]
        cache[entity] = torch.tensor([names.index(n) for n in links], device=env.device)
    return ent.data.body_link_pos_w[:, cache[entity], :2] - env.scene.env_origins[:, None, :2]


def straw_edge(radius: float) -> tuple[float, float, float]:
    """`ring_edge`'s (radius, tokudawara radius, tokudawara half-angle) for a ring
    whose bales are centred on `radius`, drawn as `env_cfg._add_ring` draws them
    (bale radius 0.07 m of a real ring, scaled; tokudawara set out a bale's width)."""
    bale = 0.07 * radius / 2.275
    half = radius * math.pi / 20 * 0.92
    return radius + bale, radius + 3 * bale, half / (radius + 2 * bale)


def ring_edge(xy: torch.Tensor, edge: tuple[float, float, float]) -> torch.Tensor:
    """The outer edge of the straw at each point's bearing: `edge` is (radius,
    radius at a tokudawara, half-angle of a tokudawara). The four tokudawara sit
    on the axes, set out by a bale's width."""
    out, toku, half = edge
    theta = torch.atan2(xy[..., 1], xy[..., 0])
    off = torch.remainder(theta + math.pi / 4, math.pi / 2) - math.pi / 4
    return torch.where(off.abs() < half, torch.full_like(theta, toku), torch.full_like(theta, out))


def stepped_out(env: "ManagerBasedRlEnv", entity: str, edge: tuple[float, float, float]) -> torch.Tensor:
    """A foot of `entity` is on the ground beyond the straw's outer edge."""
    from ..common.mdp.rewards import contact_state

    xy = _feet_xy(env, entity)
    beyond = torch.linalg.norm(xy, dim=-1) > ring_edge(xy, edge)
    return (beyond & contact_state(env, FEET_SENSOR[entity]).bool()).any(-1)


def touched_down(env: "ManagerBasedRlEnv", entity: str) -> torch.Tensor:
    """Something of `entity` other than a foot is on the ground.

    JUMPER_SUMO_DOWN_OFF=1 turns the test off (a diagnostic: to see what touches
    the ground in a bout without the bout ending on it)."""
    found = env.scene.sensors[BODY_SENSOR[entity]].data.found
    if os.environ.get("JUMPER_SUMO_DOWN_OFF") == "1":
        return torch.zeros(found.shape[0], dtype=torch.bool, device=found.device)
    return found.reshape(found.shape[0], -1).sum(-1) > 0


def out_of_ring(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg,
                edge: tuple[float, float, float]) -> torch.Tensor:
    """Termination: the entity stepped out (a foot on the ground beyond the straw),
    once the bout has begun."""
    return stepped_out(env, asset_cfg.name, edge) & bout(env).started


def down_on_clay(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Termination: something other than a foot touched the ground, once the bout
    has begun (a crouch with the claws down before the tachiai is not a loss)."""
    return touched_down(env, asset_cfg.name) & bout(env).started


def bout_outcome(env: "ManagerBasedRlEnv", terms: tuple[str, ...]) -> torch.Tensor:
    """1 where any of the named terminations fired this step, else 0."""
    tm = env.termination_manager
    fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in terms:
        fired |= tm.get_term(name).bool()
    return fired.float()


def push_out(env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg(OPPONENT),
             limit: float = 0.5) -> torch.Tensor:
    """The opponent's speed away from the ring's centre, m/s, clamped to +-limit."""
    asset = env.scene[asset_cfg.name]
    xy = asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    out = xy / torch.linalg.norm(xy, dim=-1, keepdim=True).clamp(min=1e-3)
    v = (asset.data.root_link_lin_vel_w[:, :2] * out).sum(-1)
    return v.clamp(-limit, limit)
