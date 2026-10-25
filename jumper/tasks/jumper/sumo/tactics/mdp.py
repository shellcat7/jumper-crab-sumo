"""jumper.sumo.tactics: the tactician's action, observations and rewards.

The tactician does not move joints. Its six numbers are a velocity and a posture
command, and a frozen walking policy (`TacticsAction`, the same machinery as the
opponent's) turns them into the robot's 20 joint targets. What it learns is
where to go and how to stand -- charge, side-step, circle, lean in, drop low --
against an opponent it can see:

    own        body-frame velocity (vx, vy, yaw rate), gravity in the body frame
    opponent   its position, heading, velocity and tilt, in the robot's frame
    ring       the way to the centre and how near each robot is to the straw
    bout       started or not, and for how long
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..mdp import SHIKIRI_POSTURE, bout, shikiri_or
from ..opponent import OPPONENT, OpponentPolicyAction, OpponentPolicyActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: The tactician's six outputs, each clipped to [-1, 1], mapped onto commands the
#: walking policy was trained on (posture: twist, pitch, roll, height):
#:     vx  [-0.30, 0.45] m/s    vy  +-0.30 m/s    wz  +-0.75 rad/s
#:     pitch [-0.10, 0.20] rad  roll +-0.15 rad   height [0.080, 0.130] m
COMMAND_MID = (0.075, 0.0, 0.0, 0.05, 0.0, 0.105)
COMMAND_HALF = (0.375, 0.30, 0.75, 0.15, 0.15, 0.025)
#: Commands follow the tactician's choice with this time constant, so the walking
#: policy sees a command that moves as smoothly as a stick, not a 200 Hz jitter.
SMOOTHING_S = 0.08
ACTION_DIM = 6


def action_dim() -> int:
    """Six, and the seven claw outputs on top with `JUMPER_TACTICS_WAZA=1`
    (`waza.py`)."""
    from .waza import WAZA_DIM

    return ACTION_DIM + (WAZA_DIM if os.environ.get("JUMPER_TACTICS_WAZA", "") == "1" else 0)


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _to_body(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """World-frame xy vectors [N, 2] into the frame of a body with this yaw."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1]), -1)


# ── The action ────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class TacticsActionCfg(OpponentPolicyActionCfg):
    """The robot's joints, driven by a frozen walking policy that the tactician
    steers. `checkpoint` is that walking policy."""

    def build(self, env: "ManagerBasedRlEnv") -> "TacticsAction":
        return TacticsAction(self, env)


class TacticsAction(OpponentPolicyAction):
    cfg: TacticsActionCfg

    def __init__(self, cfg: TacticsActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._mid = torch.tensor(COMMAND_MID, device=self.device)
        self._half = torch.tensor(COMMAND_HALF, device=self.device)
        #: Its own outputs: six, or thirteen with claws (`action_dim`); a frozen
        #: tactician's (rival.py) as many as its checkpoint has.
        self._dim = getattr(self, "_dim", None) or action_dim()
        self._choice = torch.zeros(self.num_envs, self._dim, device=self.device)
        self._command = self._mid.repeat(self.num_envs, 1)
        self._alpha = min(1.0, env.step_dt / SMOOTHING_S)

    @property
    def action_dim(self) -> int:
        # The joint path below still works on 20; the action manager sees 6 (13).
        return self._dim if getattr(self, "_hidden", False) else self._action_dim

    def commands(self) -> tuple[torch.Tensor, torch.Tensor]:
        state = bout(self._env)
        state.update()
        target = self._mid + self._half * self._choice[:, :ACTION_DIM]
        self._command += self._alpha * (target - self._command)
        c = self._command
        twist = torch.stack((c[:, 0], c[:, 1], c[:, 2]), -1)
        posture = torch.stack((torch.zeros_like(c[:, 3]), c[:, 3], c[:, 4], c[:, 5]), -1)
        neutral = torch.tensor([0.0, 0.0, 0.0, self.cfg.neutral_height], device=self.device)
        crouch = neutral + torch.tensor(SHIKIRI_POSTURE, device=self.device)
        from ..mdp import hold_mark

        hold = hold_mark(self._env, self.cfg.entity_name)
        return shikiri_or(twist, state.started, hold), shikiri_or(posture, state.started, crouch)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._choice = actions.clamp(-1.0, 1.0)
        if self._choice.shape[-1] > ACTION_DIM:
            from .waza import SIDES, waza

            w = waza(self._env)
            if w is not None:
                w.command(SIDES.index(self.cfg.entity_name), self._choice[:, ACTION_DIM:])
        twist, posture = self.commands()
        self._env.command_manager.get_term(self.cfg.twist_name).set(twist)
        self._env.command_manager.get_term(self.cfg.posture_name).set(posture)
        # Skip OpponentPolicyAction.process_actions: the joint path, as it does.
        super(OpponentPolicyAction, self).process_actions(self._act(self._observe()))
        self._overlay_arms()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._command[ids] = self._mid
        self._choice[ids] = 0.0
        if self.cfg.entity_name == "robot":
            from .waza import waza

            w = waza(self._env)
            if w is not None:
                ids_t = (torch.arange(self.num_envs, device=self.device) if env_ids is None
                         else torch.as_tensor(env_ids, device=self.device))
                w.reset(ids_t)


# ── Observations ──────────────────────────────────────────────────────────────


# Each term takes `me` and `other` (entity names): the robot's tactician sees from
# the robot, and a frozen tactician driving the opponent (`rival.py`) from the
# opponent, with the same functions.


def own_motion(env: "ManagerBasedRlEnv", me: str = "robot") -> torch.Tensor:
    """[vx, vy, yaw rate] in the body frame, and gravity in the body frame (6)."""
    d = env.scene[me].data
    return torch.cat((d.root_link_lin_vel_b[:, :2], d.root_link_ang_vel_b[:, 2:3],
                      d.projected_gravity_b), -1)


def opponent_relative(env: "ManagerBasedRlEnv", me: str = "robot", other: str = OPPONENT) -> torch.Tensor:
    """The opponent seen from the robot (8): position (x, y) and distance; its
    heading relative to the robot's (cos, sin); its velocity (x, y); its tilt
    (the z of gravity in its body frame, -1 upright)."""
    me, op = env.scene[me].data, env.scene[other].data
    yaw = _yaw(me.root_link_quat_w)
    rel = _to_body(op.root_link_pos_w[:, :2] - me.root_link_pos_w[:, :2], yaw)
    dist = torch.linalg.norm(rel, dim=-1, keepdim=True)
    dyaw = _yaw(op.root_link_quat_w) - yaw
    vel = _to_body(op.root_link_lin_vel_w[:, :2], yaw)
    return torch.cat((rel, dist, torch.cos(dyaw)[:, None], torch.sin(dyaw)[:, None], vel,
                      op.projected_gravity_b[:, 2:3]), -1)


def opponent_trend(env: "ManagerBasedRlEnv", tau: float = 0.5, me: str = "robot",
                   other: str = OPPONENT) -> torch.Tensor:
    """How the opponent has been moving lately (3): its velocity in the robot's
    frame and the speed the two have been closing at, each a moving average over
    about `tau` seconds. The current velocity says what it is doing; this says
    what kind of wrestler it is -- one that charges, waits or circles -- which a
    memoryless policy cannot otherwise tell when the opponent changes from bout to
    bout (the opponent pool). Kept once per environment step on the env, so the
    actor's and the critic's copies of the term read the same average.
    """
    key = f"_sumo_trend_{me}"
    me, op = env.scene[me].data, env.scene[other].data
    yaw = _yaw(me.root_link_quat_w)
    vel = _to_body(op.root_link_lin_vel_w[:, :2], yaw)
    rel = op.root_link_pos_w[:, :2] - me.root_link_pos_w[:, :2]
    dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-3)
    closing = -((op.root_link_lin_vel_w[:, :2] - me.root_link_lin_vel_w[:, :2]) * rel).sum(-1) / dist
    x = torch.cat((vel, closing[:, None]), -1)
    step = int(env.common_step_counter)
    state = getattr(env, key, None)
    if state is None or state["ema"].shape != x.shape:
        state = {"step": step, "ema": x.clone()}
        setattr(env, key, state)
    elif state["step"] != step:
        fresh = (env.episode_length_buf <= 1)[:, None]
        a = min(1.0, env.step_dt / tau)
        state["ema"] = torch.where(fresh, x, state["ema"] + a * (x - state["ema"]))
        state["step"] = step
    return state["ema"]


#: Memory (`JUMPER_TACTICS_MEMORY=1`, 10/14 night): a technique is a sequence -- unbalance,
#: take hold, throw -- and a policy that sees only the present cannot know which step
#: of it it is in (the trend's moving average is the only past it had). The last
#: `MEMORY_FRAMES` snapshots, `MEMORY_STRIDE_S` apart (about a second), are shown.
MEMORY_FRAMES = int(os.environ.get("JUMPER_TACTICS_MEMORY_FRAMES", "8"))
MEMORY_STRIDE_S = float(os.environ.get("JUMPER_TACTICS_MEMORY_STRIDE_S", "0.125"))
MEMORY_DIM = 12          # per frame: the opponent (8), its own velocity (3), touching (1)


def bout_memory(env: "ManagerBasedRlEnv", me: str = "robot", other: str = OPPONENT) -> torch.Tensor:
    """The bout as it was (`MEMORY_FRAMES` x 12, oldest first): each frame the
    opponent as `opponent_relative` saw it then, its own velocity (vx, vy, yaw rate)
    and whether the two were touching. Kept on the env once per step, so the actor's
    and critic's copies read one buffer; a fresh episode fills it with the present."""
    from ..referee import _found

    key = f"_sumo_memory_{me}"
    touching = _found(env, "bout_contact").float()[:, None]
    snap = torch.cat((opponent_relative(env, me, other), own_motion(env, me)[:, :3], touching), -1)
    step = int(env.common_step_counter)
    stride = max(1, round(MEMORY_STRIDE_S / env.step_dt))
    state = getattr(env, key, None)
    if state is None or state["buf"].shape[0] != snap.shape[0]:
        state = {"step": step, "buf": snap[:, None, :].repeat(1, MEMORY_FRAMES, 1)}
        setattr(env, key, state)
    elif state["step"] != step:
        buf = state["buf"]
        if step % stride == 0:
            buf = torch.cat((buf[:, 1:], snap[:, None]), 1)
        fresh = (env.episode_length_buf <= 1)[:, None, None]
        state["buf"] = torch.where(fresh, snap[:, None, :].expand_as(buf), buf)
        state["step"] = step
    return state["buf"].reshape(snap.shape[0], -1)


def ring_state(env: "ManagerBasedRlEnv", radius: float, me: str = "robot",
               other: str = OPPONENT) -> torch.Tensor:
    """The ring seen from the robot (6): the way to the centre (x, y, robot frame),
    each robot's distance from the centre as a fraction of the radius, and the
    direction that would push the opponent out (x, y, robot frame)."""
    me, op = env.scene[me].data, env.scene[other].data
    origin = env.scene.env_origins[:, :2]
    yaw = _yaw(me.root_link_quat_w)
    mine = me.root_link_pos_w[:, :2] - origin
    theirs = op.root_link_pos_w[:, :2] - origin
    r_me = torch.linalg.norm(mine, dim=-1, keepdim=True)
    r_op = torch.linalg.norm(theirs, dim=-1, keepdim=True)
    to_centre = _to_body(-mine, yaw)
    out = _to_body(theirs / r_op.clamp(min=1e-3), yaw)
    return torch.cat((to_centre, r_me / radius, r_op / radius, out), -1)


def bout_phase(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """[started, seconds since the tachiai / 10] (2)."""
    state = bout(env)
    since = torch.where(state.started,
                        (env.common_step_counter - state.start_step).float() * env.step_dt / 10.0,
                        torch.zeros_like(state.elapsed))
    return torch.stack((state.started.float(), since.clamp(max=2.0)), -1)


def in_contact(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """1 while any part of the robot touches the opponent (critic only)."""
    found = env.scene.sensors["bout_contact"].data.found
    return (found.reshape(found.shape[0], -1).sum(-1) > 0).float()[:, None]


# ── Rewards ───────────────────────────────────────────────────────────────────


def near_straw(env: "ManagerBasedRlEnv", radius: float, margin: float = 0.25) -> torch.Tensor:
    """0 inside `radius - margin`, rising to 1 at the straw (squared)."""
    d = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    depth = (torch.linalg.norm(d, dim=-1) - (radius - margin)).clamp(min=0.0) / margin
    return depth.square()


def behind(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """While the two are touching after the tachiai: how far round the opponent
    the robot is, 0 straight in front of it to 1 directly behind (the bearing of
    the robot from the opponent's heading, / pi). A style term: it pays for
    working to the side and back -- where okuridashi comes from -- which a
    bonus paid only on an okuridashi win never gets to teach, since a policy that
    wins by pushing straight never finds one (the coach run of 10/10)."""
    from ..referee import _found

    state = bout(env)
    me, op = env.scene["robot"].data, env.scene[OPPONENT].data
    rel = me.root_link_pos_w[:, :2] - op.root_link_pos_w[:, :2]
    bearing = torch.atan2(rel[:, 1], rel[:, 0]) - _yaw(op.root_link_quat_w)
    bearing = torch.remainder(bearing + math.pi, 2 * math.pi) - math.pi
    touching = _found(env, "bout_contact")
    return torch.where(state.started & touching, bearing.abs() / math.pi, torch.zeros_like(bearing))


def kimarite_won(env: "ManagerBasedRlEnv", names: tuple[str, ...], radius: float) -> torch.Tensor:
    """1 on the step the referee gives the robot (east) a win by one of `names`
    (romaji, `referee.KIMARITE`): a technique coach's bonus."""
    from ..referee import KIMARITE, referee

    r = referee(env, radius)
    r.update()
    wanted = torch.tensor([i for i, (n, _) in enumerate(KIMARITE) if n in names], device=env.device)
    now = r.decided & (r.decided_at == r.time) & (r.winner == 0)
    return (now & torch.isin(r.kimarite, wanted)).float()


#: Variety (`JUMPER_TACTICS_VARIETY=bonus`, 10/20): the leagues' wrestlers win 85-90 % of
#: their bouts by oshidashi, whatever the claws can do -- a push is the surest win and the
#: win's 30 is all a technique earns. Here a win also pays `bonus` x (1 - its kimarite's
#: share of the robot's recent wins): an oshidashi (share 0.85) earns a little, a yorikiri,
#: a throw or a hold (a few %) nearly all of it, and as a technique catches on its share
#: rises and its bonus falls, so no single other one takes over. The share is a moving
#: histogram over about `VARIETY_WINS` wins, across all environments. A win by the
#: opponent's own fault (isamiashi, koshikudake) or a dotai earns no bonus.
VARIETY_WINS = 4000.0
_NO_VARIETY = ("", "isamiashi", "koshikudake", "dotai")


def rare_kimarite_won(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """1 - share on the step the robot (east) wins by a kimarite (see VARIETY_WINS)."""
    from ..referee import KIMARITE, referee

    r = referee(env, radius)
    r.update()
    nk = len(KIMARITE)
    state = getattr(env, "_sumo_kimarite_share", None)
    if state is None:
        eligible = torch.tensor([n not in _NO_VARIETY for n, _ in KIMARITE], device=env.device)
        share = torch.zeros(nk, device=env.device)
        share[_K_OSHIDASHI] = 1.0               # it starts as it is: nearly all pushes
        state = {"step": -1, "share": share, "eligible": eligible, "pay": torch.zeros(r.decided.shape[0], device=env.device)}
        setattr(env, "_sumo_kimarite_share", state)
    step = int(env.common_step_counter)
    if state["step"] != step:                   # once per step: actor and critic read one
        now = r.decided & (r.decided_at == r.time) & (r.winner == 0)
        k = r.kimarite.clamp(0, nk - 1)
        pay = torch.where(now & state["eligible"][k], 1.0 - state["share"][k], torch.zeros_like(state["pay"]))
        wins = now & state["eligible"][k]
        n = int(wins.sum())
        if n:
            h = torch.bincount(k[wins], minlength=nk).float() / n
            a = min(1.0, n / VARIETY_WINS)
            state["share"] = (1 - a) * state["share"] + a * h
        state["pay"], state["step"] = pay, step
    return state["pay"]


_K_OSHIDASHI = 1      # referee.KIMARITE: ("oshidashi", ...) is index 1


#: Comebacks (`JUMPER_TACTICS_COMEBACK=bonus`, 10/21): the leagues' bouts are mostly one
#: wrestler walking the other back and out, and a wrestler driven to the straw has learned to
#: expect the loss. A win also pays `bonus` x how deep the winner had been driven before it --
#: 0 if it never came within `COMEBACK_MARGIN` of the straw after the tachiai, 1 if its base
#: reached the straw itself -- so holding at the edge and turning the bout (an utchari, a
#: pull-down, a sidestep) pays, the bouts the crowd rises for. A win by the opponent's own
#: fault (isamiashi, koshikudake) or a dotai earns no bonus.
COMEBACK_MARGIN = 0.2


def comeback_won(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """On the step the robot (east) wins: how deep it had been driven towards the straw
    (0..1, see COMEBACK_MARGIN)."""
    from ..referee import KIMARITE, referee

    r = referee(env, radius)
    r.update()
    state = getattr(env, "_sumo_comeback", None)
    n = r.decided.shape[0]
    if state is None:
        eligible = torch.tensor([k not in _NO_VARIETY for k, _ in KIMARITE], device=env.device)
        state = {"step": -1, "deep": torch.zeros(n, device=env.device), "eligible": eligible,
                 "pay": torch.zeros(n, device=env.device)}
        setattr(env, "_sumo_comeback", state)
    step = int(env.common_step_counter)
    if state["step"] != step:                   # once per step
        started = bout(env).started
        d = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
        depth = ((torch.linalg.norm(d, dim=-1) - (radius - COMEBACK_MARGIN)) / COMEBACK_MARGIN).clamp(0.0, 1.0)
        live = started & ~r.decided
        state["deep"] = torch.where(started, torch.where(live, torch.maximum(state["deep"], depth), state["deep"]),
                                    torch.zeros_like(depth))
        now = r.decided & (r.decided_at == r.time) & (r.winner == 0)
        k = r.kimarite.clamp(0, len(KIMARITE) - 1)
        state["pay"] = torch.where(now & state["eligible"][k], state["deep"], torch.zeros_like(depth))
        state["step"] = step
    return state["pay"]


def closing(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """After the tachiai, -distance to the opponent beyond 0.35 m: no prize for
    running away and waiting out the clock."""
    state = bout(env)
    me, op = env.scene["robot"].data, env.scene[OPPONENT].data
    d = torch.linalg.norm(op.root_link_pos_w[:, :2] - me.root_link_pos_w[:, :2], dim=-1)
    return torch.where(state.started, -(d - 0.35).clamp(min=0.0), torch.zeros_like(d))


__all__ = [
    "ACTION_DIM", "TacticsAction", "action_dim", "TacticsActionCfg", "behind", "bout_phase", "closing",
    "comeback_won", "in_contact", "kimarite_won", "near_straw", "opponent_trend", "opponent_relative", "own_motion", "rare_kimarite_won",
    "ring_state",
]
