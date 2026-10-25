"""The gyoji: a third Jumper in the ring that referees the bout (jumper-challenge-2026).

It is a real body, not a picture: it collides with both wrestlers, so it can be
bumped into or knocked over as a gyoji sometimes is. It is driven like the
opponent (`opponent.OpponentPolicyAction`) by a frozen walking policy; what it is
told to do comes from `GyojiPolicyAction.commands`:

- stand off the line between the two wrestlers, on the side it took in the
  shikiri, at `WATCH_OFFSET` from their midpoint and inside the straw -- the side
  view a gyoji keeps -- turning to face them;
- when the bout drifts to the straw on its side, go round to the other side,
  circling the pair out of reach rather than cutting between them;
- step away from either wrestler that comes within `KEEP_CLEAR`, onto or past the
  straw if need be (it can be too slow, and then it is hit);
- after the decision, stop and turn to the winner.

Its judgement is `referee.Referee`; its shell is
recoloured purple and its gunbai drawn in the decoration group only, so none of
that touches the physics. The gunbai is in its right claw, which it carries up in front of it and
raises to call the winner (`arms.py`); the gyoji walks on the other five legs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from .opponent import OpponentPolicyAction, OpponentPolicyActionCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

GYOJI = "gyoji"
TWIST_GYOJI = "twist_gyoji"
POSTURE_GYOJI = "posture_gyoji"
#: Spacing (10/10 evening, the user: the gyoji was too close, claws and bodies
#: touching). A wrestler's claws reach about 0.2 m ahead of its base and the
#: gyoji's gunbai arm about as far, so bases 0.5 m apart leave about 10 cm.
WATCH_OFFSET = 0.55      # m from the wrestlers' midpoint, across their line
KEEP_CLEAR = 0.50        # m from either wrestler's base
INSIDE = 0.95            # m from the centre at most: onto the mound beyond the straw
INSIDE_SOFT = 0.85       # m: beyond this (just past the straw) it is drawn back in
MAX_SPEED = 0.4          # m/s
#: Where it stands and how it moves round (10/11, the user: a gyoji's real moves,
#: going round the wrestlers). Its spot is kept inside the straw (`STAY_INSIDE`,
#: the straw at 0.8 m); when the bout drifts to the straw on its side, so that its
#: spot is beyond `SWITCH_OUT` and the other side's is `SWITCH_GAIN` nearer the centre,
#: it goes round to the other side -- circling the pair out of reach when its
#: spot is more than `ORBIT_ANGLE` round from where it stands, never between them.
STAY_INSIDE = 0.62       # m
SWITCH_OUT = 0.55        # m: its spot this far out or more, ...
SWITCH_GAIN = 0.10       # m: ... and the other side's this much nearer the centre
ORBIT_ANGLE = 0.7        # rad
#: Close enough to its spot, and facing the bout closely enough, to stand still:
#: within these it is told nothing, and its legs are held (`_overlay_arms`).
#: Without them a walking policy steps in place at the smallest command, and with
#: the gunbai up on five legs every step bobs it 2 cm (10/10, through the shikiri).
STILL_DIST = 0.03        # m
STILL_YAW = 0.15         # rad


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


@dataclass(kw_only=True)
class GyojiPolicyActionCfg(OpponentPolicyActionCfg):
    radius: float = 0.8

    def build(self, env: "ManagerBasedRlEnv") -> "GyojiPolicyAction":
        return GyojiPolicyAction(self, env)


class GyojiPolicyAction(OpponentPolicyAction):
    cfg: GyojiPolicyActionCfg

    def __init__(self, cfg: GyojiPolicyActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        from .mdp import bout
        from .referee import referee

        # The gyoji, not a timer, starts the bout: no start without the hands down.
        referee(env, cfg.radius).officiates = True
        bout(env).max_shikiri_s = float("inf")
        # ... and checks the hands are behind the shikiri lines (inner edge: half
        # the 70 cm gap, scaled as env_cfg._add_ring draws it).
        bout(env).line_x = 0.70 * cfg.radius / 2.275 / 2
        import os

        if os.environ.get("JUMPER_SUMO_AUDIENCE") == "1":
            # The hall's judges and spectators move with the bout (audience.py):
            # the renderer calls this once per frame.
            from .audience import Crowd

            env.sumo_decorate = Crowd(env)

    def _overlay_arms(self) -> None:
        """The gunbai in the right claw: held up in front, raised to call the winner.
        Standing still (see `commands`), the legs are held at the stance, eased in
        and out over about 50 ms so starting to walk again is not a jolt."""
        from .arms import GYOJI_HOLD, GYOJI_RAISE, STANCE, overlay
        from .referee import referee

        ones = torch.ones(self.num_envs, device=self.device)
        still = getattr(self, "_still", None)
        if still is not None:
            w = getattr(self, "_stance_w", torch.zeros(self.num_envs, device=self.device))
            self._stance_w = w + 0.1 * (still.float() - w)
            overlay(self, STANCE, self._stance_w)
        overlay(self, GYOJI_HOLD, ones)
        overlay(self, GYOJI_RAISE, referee(self._env, self.cfg.radius).decided.float())

    def commands(self) -> tuple[torch.Tensor, torch.Tensor]:
        from .referee import referee

        env = self._env
        ref = referee(env, self.cfg.radius)
        ref.update()
        origin = env.scene.env_origins[:, :2]
        a = env.scene["robot"].data.root_link_pos_w[:, :2]
        b = env.scene["opponent"].data.root_link_pos_w[:, :2]
        me = self._entity.data
        g = me.root_link_pos_w[:, :2]
        yaw = _yaw(me.root_link_quat_w)

        from .mdp import bout

        waiting = ~bout(env).started
        mid = (a + b) / 2
        d = b - a
        sep = torch.linalg.norm(d, dim=-1, keepdim=True).clamp(min=1e-3)
        d = d / sep
        n = torch.stack((-d[:, 1], d[:, 0]), -1)
        # Its side of the wrestlers' line: where it stands in the shikiri, kept
        # through the bout -- until the bout drifts to the straw on that side, and
        # then it goes round to the other (with some hysteresis, so it does not
        # dither between the two).
        here = torch.sign(((g - mid) * n).sum(-1))
        here = torch.where(here == 0, torch.ones_like(here), here)
        side = getattr(self, "_side", None)
        side = here if side is None else torch.where(waiting, here, side)
        r_here = torch.linalg.norm(mid + side[:, None] * n * WATCH_OFFSET - origin, dim=-1)
        r_there = torch.linalg.norm(mid - side[:, None] * n * WATCH_OFFSET - origin, dim=-1)
        switch = ~waiting & ~ref.decided & (r_here > SWITCH_OUT) & (r_there < r_here - SWITCH_GAIN)
        side = torch.where(switch, -side, side)
        self._side = side
        goal = mid + side[:, None] * n * WATCH_OFFSET
        # Inside the straw by preference; the wrestlers' keep-clear may still push
        # it over, as a real gyoji steps onto or past the straw to make room.
        rel = goal - origin
        goal = origin + rel * (STAY_INSIDE / torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-3)).clamp(max=1.0)
        for w in (a, b):
            diff = g - w
            dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
            goal = goal + diff / dist * (KEEP_CLEAR - dist).clamp(min=0.0) * 3.0
        v = (goal - g) * 1.5
        # Going round: when its spot is on the far side of the pair, it does not
        # cut between them but circles them, out of reach, the shorter way.
        u = g - mid
        ru = torch.linalg.norm(u, dim=-1, keepdim=True).clamp(min=1e-3)
        wv = goal - mid
        ang = torch.atan2(u[:, 0] * wv[:, 1] - u[:, 1] * wv[:, 0], (u * wv).sum(-1))
        tangent = torch.stack((-u[:, 1], u[:, 0]), -1) / ru * torch.sign(ang)[:, None]
        orbit_r = torch.maximum(torch.full_like(ru, WATCH_OFFSET), KEEP_CLEAR + 0.5 * sep)
        orbit = tangent * MAX_SPEED + (orbit_r - ru) * u / ru * 2.0
        v = torch.where((ang.abs() > ORBIT_ANGLE)[:, None], orbit, v)
        # Out of reach whatever it is doing: going round, too, it steps away from a
        # wrestler inside `KEEP_CLEAR` (without this, going round brought it into
        # them: 223 steps touching in the test of 10/11).
        for w in (a, b):
            diff = g - w
            dist = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=1e-3)
            v = v + diff / dist * (KEEP_CLEAR - dist).clamp(min=0.0) * 6.0
        # Never off the mound's top.
        rg = g - origin
        r = torch.linalg.norm(rg, dim=-1, keepdim=True).clamp(min=1e-3)
        v = v - rg / r * (r - INSIDE_SOFT).clamp(min=0.0) * 4.0
        v = v * (MAX_SPEED / torch.linalg.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)).clamp(max=1.0)
        # Watch the midpoint; after the decision, stand and face the winner.
        look = torch.where((ref.decided & (ref.winner >= 0))[:, None],
                           torch.where((ref.winner == 0)[:, None], a, b), mid)
        # Still through the shikiri (a gyoji stands with the gunbai up while the
        # wrestlers settle) and after the decision; in the bout, still when close
        # to its spot.
        near = torch.linalg.norm(goal - g, dim=-1) < STILL_DIST
        v = torch.where((ref.decided | waiting | near)[:, None], torch.zeros_like(v), v)
        err = _wrap(torch.atan2(look[:, 1] - g[:, 1], look[:, 0] - g[:, 0]) - yaw)
        wz = torch.clamp(2.0 * err, -0.75, 0.75)
        wz = torch.where(waiting | (err.abs() < STILL_YAW), torch.zeros_like(wz), wz)
        self._still = (torch.linalg.norm(v, dim=-1) == 0) & (wz == 0)
        c, s = torch.cos(yaw), torch.sin(yaw)
        twist = torch.stack((c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], wz), -1)
        posture = torch.tensor([0.0, 0.0, 0.0, self.cfg.neutral_height], device=self.device)
        return twist, posture.expand(self.num_envs, 4)


#: Where the gunbai sits in the gyoji's right claw, in the frame of its palm pad
#: (`RF_palm_pad_b_link`): the grip point between the pad and the finger tip.
#: Worked out at `arms.GYOJI_HOLD` (10/10): the quaternion turns the gunbai's own
#: axes (x across the blade, y through it, z up the handle) to that pose's up and
#: forward.
GUNBAI_QUAT = [0.3469, 0.6783, 0.0828, 0.6424]
GUNBAI_GRIP = [0.0152, -0.0053, -0.0063]

#: The gunbai, in its own frame (m, z up from the grip). A real gunbai's blade is
#: egg- or gourd-shaped -- broad and round at the top, narrowing to a neck where a
#: metal collar joins it to the handle -- lacquered, with a gilt rim and a crest,
#: and a tassel hangs from the end of the handle. (Until 10/10 evening: two
#: stacked ellipsoids and a capsule; the user asked for a truer shape.)
BLADE_BOTTOM, BLADE_TOP = 0.018, 0.118   # neck to crown
BLADE_HALF_WIDTH = 0.040                 # before the egg's swell (EGG)
EGG = 0.28                               # how much wider the top is than the bottom
BLADE_HALF_THICK, RIM_HALF_THICK = 0.0022, 0.0015
LACQUER, GILT, CREST, HANDLE, TASSEL = ((0.07, 0.05, 0.05, 1.0), (0.86, 0.70, 0.28, 1.0),
                                        (0.70, 0.08, 0.08, 1.0), (0.20, 0.10, 0.06, 1.0),
                                        (0.45, 0.18, 0.60, 1.0))


def _outline(scale: float = 1.0, n: int = 48) -> list[tuple[float, float]]:
    """The blade's outline (x, z): an egg, its narrow end down at the neck."""
    zc = (BLADE_BOTTOM + BLADE_TOP) / 2
    b = (BLADE_TOP - BLADE_BOTTOM) / 2
    pts = []
    for i in range(n):
        t = 2 * math.pi * i / n                    # 0 at the neck, pi at the crown
        x = BLADE_HALF_WIDTH * math.sin(t) * (1 - EGG * math.cos(t))
        z = zc - b * math.cos(t)
        pts.append((x * scale, zc + (z - zc) * scale))
    return pts


def _slab(spec, name: str, outline: list[tuple[float, float]], half_thick: float) -> str:
    """A flat plate with this outline, `half_thick` either side of y = 0, as a mesh."""
    n = len(outline)
    cx = sum(x for x, _ in outline) / n
    cz = sum(z for _, z in outline) / n
    verts = [(cx, half_thick, cz), (cx, -half_thick, cz)]
    verts += [(x, half_thick, z) for x, z in outline] + [(x, -half_thick, z) for x, z in outline]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        f0, f1, b0, b1 = 2 + i, 2 + j, 2 + n + i, 2 + n + j
        faces += [(0, f1, f0), (1, b0, b1), (f0, f1, b1), (f0, b1, b0)]
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = [c for v in verts for c in v]
    mesh.userface = [k for f in faces for k in f]
    return name


def _gunbai(spec, hand) -> None:
    """Blade, gilt rim and crest, collar, handle, end cap and tassel, placed in the
    gyoji's claw (`GUNBAI_GRIP`, `GUNBAI_QUAT`); decoration group, no mass."""
    from .env_cfg import DECOR_GROUP

    vis = dict(contype=0, conaffinity=0, mass=0.0, group=DECOR_GROUP)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, np.array(GUNBAI_QUAT))
    rot = rot.reshape(3, 3)

    def at(local):  # a point of the gunbai's frame, in the claw's
        return list(np.array(GUNBAI_GRIP) + rot @ np.array(local))

    def axis_quat(direction):  # a capsule or cylinder along this gunbai-frame axis
        z = rot @ np.array(direction, dtype=float)
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, z)
        return list(q)

    zc = (BLADE_BOTTOM + BLADE_TOP) / 2
    blade = _slab(spec, "gunbai_blade_mesh", _outline(), BLADE_HALF_THICK)
    rim = _slab(spec, "gunbai_rim_mesh", _outline(1.09), RIM_HALF_THICK)
    hand.add_geom(name="gunbai_blade", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=blade,
                  pos=GUNBAI_GRIP, quat=GUNBAI_QUAT, rgba=list(LACQUER), **vis)
    hand.add_geom(name="gunbai_rim", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=rim,
                  pos=GUNBAI_GRIP, quat=GUNBAI_QUAT, rgba=list(GILT), **vis)
    # The crest: a gilt ring round a red disc, on both faces.
    through = axis_quat((0, 1, 0))
    hand.add_geom(name="gunbai_crest_ring", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[0.013, BLADE_HALF_THICK + 0.0004, 0], pos=at((0, 0, zc + 0.008)),
                  quat=through, rgba=list(GILT), **vis)
    hand.add_geom(name="gunbai_crest", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[0.009, BLADE_HALF_THICK + 0.0007, 0], pos=at((0, 0, zc + 0.008)),
                  quat=through, rgba=list(CREST), **vis)
    up = axis_quat((0, 0, 1))
    hand.add_geom(name="gunbai_collar", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[0.0065, 0.004, 0], pos=at((0, 0, BLADE_BOTTOM - 0.002)),
                  quat=up, rgba=list(GILT), **vis)
    hand.add_geom(name="gunbai_handle", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=[0.0045, 0.022, 0], pos=at((0, 0, -0.008)), quat=up,
                  rgba=list(HANDLE), **vis)
    hand.add_geom(name="gunbai_end", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[0.0058, 0, 0], pos=at((0, 0, -0.032)), rgba=list(GILT), **vis)
    # The tassel: a cord from the knot, then the fringe, wider, hanging below.
    hand.add_geom(name="gunbai_tassel_cord", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=[0.0028, 0.007, 0], pos=at((0, 0, -0.046)), quat=up,
                  rgba=list(TASSEL), **vis)
    hand.add_geom(name="gunbai_tassel", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[0.0075, 0.010, 0], pos=at((0, 0, -0.064)), quat=up,
                  rgba=list(TASSEL), **vis)
    hand.add_geom(name="gunbai_tassel_knot", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[0.0052, 0, 0], pos=at((0, 0, -0.037)), rgba=list(TASSEL), **vis)


def _dress(spec) -> None:
    """The gyoji's shell in purple, and the gunbai in its right claw.

    Its own parts recoloured, as a skin would: the red upper shell purple. (Until
    10/10 evening it wore an added robe dome, collar and eboshi; the user asked for
    the robot's real parts recoloured instead.) The gunbai is a prop it carries,
    in the decoration group: no mass, no contact.
    """
    from .env_cfg import recolor

    recolor(spec, f"{GYOJI}/upper_shell_link_visual", (0.45, 0.18, 0.60, 1.0))
    hand = next((b for b in spec.bodies if b.name.endswith(f"{GYOJI}/RF_palm_pad_b_link")), None)
    if hand is not None:
        _gunbai(spec, hand)
