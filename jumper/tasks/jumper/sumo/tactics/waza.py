"""Waza: what a wrestler does with its claws, and how it stands (jumper-challenge-2026).

Until 10/10 a Jumper wrestled with its body and legs only: its claws were front
feet, held closed, and the tactician could choose where to go and how low to
stand, nothing more. Real sumo is two families of wrestling (docs/notes/
rikishi-and-waza.md): *oshi-zumo*, apart, thrusting and slapping (tsuki, harite,
and at the tachiai kachiage, the forearm driven up into the chest); and
*yotsu-zumo*, close, a hand on the opponent's mawashi -- from where come the
force-outs (yorikiri) and the throws (uwatenage, shitatenage); and the tricks
between, the slap down (hatakikomi), the leg pick (ashitori), the arm pull
(tottari) and the side-step at the tachiai (henka).

This module gives a tactician the claws. Seven more outputs, each in [-1, 1]:

    claw L, claw R     above 0.2: "hold" -- the claw opens and the arm reaches
                       forward (`REACH`, high or low, below), and the moment its
                       tip is within reach of something of the opponent's it can
                       hold, it closes and holds it (a grip, below); below -0.3 it
                       lets go; between, a grip is kept. (Until 10/11 the policy
                       had to close the claw itself at the right moment: no run
                       found that by chance, and the first trained wrestler with
                       claws never used them.)
    strike L, strike R above 0.5: a strike with that arm, if it is free
    kind               a strike's kind: below -0.33 hataki (slap down), above
                       0.33 kachiage (up), between a thrust -- a tsuki with the
                       claw closed, a harite (sweeping open-claw slap) with it
                       open. While reaching: below 0 reach low, for the legs
    power              the strike's speed: 0.34 s (-1) to 0.16 s (+1) end to end
    stance             how deep the shikiri crouch is: 0 and below the usual
                       stance, 1 the low one (`arms.STANCE_LOW`)

**What a crab holds.** A mawashi is a person's; a Jumper has none, and a belt as
wide as its body is nothing a claw can close round. What a crab's pincer closes
on is the other crab (the user, 10/10 night): its legs and arms -- thin links a
claw really can pinch -- and the rim of its shell. So a claw holds one of
`TARGETS`: the shell (the base and shell, as a box: holding it is what a mawashi
grip is for -- force-outs, throws), a leg segment (thigh or calf: the leg pick,
ashitori) or an arm segment (upper arm or forearm: the arm pull, tottari).

A grip is a spring between the claw tip and the point it closed on: at most
`GRIP_MAX` (12 N, about what one 1.75 N m arm servo holds at the claw's reach),
cut when stretched past `GRIP_BREAK`. It is applied as equal and opposite forces
(`xfrc_applied`): on the held body at the held point -- so a held leg is pulled
off the clay -- and on the holder's base at its claw tip, the arm's pull on its
body. It pulls, turns and trips through the physics like any other force.

**The strikes** are arm trajectories laid over the walking policy's joint
targets (`arms.overlay`), from a wind-up pose to a finish pose found by searching
the claw tip's position over the arm's joints (10/10 night); what they do to the
opponent is the contact physics.

Both wrestlers' claws live in one object on the environment (`waza(env)`): each
side's action term hands it its commands, it updates once per step, applies the
grip forces and lays the arm poses over each side's targets. A side whose policy
has no claw outputs (every walker, every tactician trained before this) keeps
them closed and on the clay, as before.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

WAZA_DIM = 7
CLAWS_OBS_DIM = 22
SIDES = ("robot", "opponent")
#: The four kinds of wrestler a waza tactician is trained as, given to it as a
#: one-hot input and paid by kind (the tables below): oshi (thrust and push),
#: yotsu (hold the shell, force out), nage (hold the shell, throw), waza (tricks:
#: slap down, leg pick, arm pull).
STYLES = ("oshi", "yotsu", "nage", "waza")
#: What each kind does unless its policy says otherwise, added to its seven claw
#: outputs (claw L, claw R, strike L, strike R, kind, power, stance). Without it,
#: three runs of 10/10-11 never found a grip: a reach that must be held for a
#: tenth of a second to get anywhere is not something per-step noise stumbles on.
STYLE_LEAN = (
    (0.0, 0.0, 0.8, 0.8, 0.0, 0.3, 0.0),     # oshi: strikes (thrust; harite when open)
    (0.7, 0.6, 0.0, 0.0, 0.5, 0.0, 0.0),     # yotsu: reach high for the shell
    (0.7, 0.6, 0.0, 0.0, 0.5, 0.0, 0.0),     # nage: the same hold, paid for turning
    (0.7, 0.6, 0.4, 0.4, -0.8, 0.0, 0.0),    # waza: reach low for the legs; hataki
)

#: Claw J4 fully open, rad (left negative, right positive; 0 is closed, HOME).
CLAW_OPEN = 1.2
#: Left arm (J0..J3); the right arm mirrors J0, J2, J3.
REACH = (-0.6, -2.0, 0.0, -1.3)          # claw tip 23 cm ahead at the body's height: the shell
REACH_LOW = (-0.8, -1.2, 0.0, -0.95)     # 26 cm ahead, 7 cm lower: the legs
STRIKE_POSES = {                         # (wind-up, finish)
    "tsuki": ((-1.6, -3.6, 0.0, -2.0), (-1.6, -3.4, 0.0, -0.6)),
    "harite": ((0.0, -2.6, 0.0, -0.95), (-1.6, -2.6, 0.0, -0.95)),
    "hataki": ((-1.4, -3.2, 0.0, -1.65), (-0.8, -1.4, 0.0, -1.3)),
    "kachiage": ((-0.4, -1.4, 0.0, -2.0), (-1.6, -3.2, 0.0, -1.3)),
}
KINDS = ("tsuki", "harite", "hataki", "kachiage")
STRIKE_FAST_S, STRIKE_SLOW_S = 0.16, 0.34
STRIKE_COOL_S = 0.25
#: The claw outputs are smoothed with this time constant before they act, as the
#: tactician's commands are (`mdp.SMOOTHING_S`). Unsmoothed, a policy exploring
#: with noise on all seven threw a strike or lifted a claw off the clay every few
#: steps: the first waza run (Colab, 10/10 night) went from winning nearly every
#: bout to losing nearly every one, unable to stand on its front claws.
WAZA_SMOOTHING_S = 0.04
#: Claws reach and strike only when the opponent is this close (base to base, m):
#: there is nothing to hold or hit further off, and a claw lifted off the clay
#: then is only a front foot lost (the smoothed run still lost two bouts in three).
#: `JUMPER_WAZA_RANGE` sets it: at 0.45 a claw lifts some 12 cm before the
#: shells meet (the reach puts its tip 23 cm ahead); the strongest claw
#: wrestlers of 10/13 gave the claws up, a front foot being worth more.
WAZA_RANGE = float(os.environ.get("JUMPER_WAZA_RANGE", "0.45"))
#: ... and only when it is in front: within this angle of the wrestler's heading
#: (rad; `JUMPER_WAZA_ARC`, degrees, 180 for no limit). The claws reach and
#: strike forward; with the opponent at the side or behind there is nothing ahead
#: to hold, and a claw lifted then is a front foot given up to a wrestler already
#: going round. Without it every claw wrestler lost to 回ノ里 (tactics_mawari, the
#: one that goes round) about three bouts in four, from either side, where 総ノ山
#: without claws broke even (the round robin, 10/12).
WAZA_ARC = float(os.environ.get("JUMPER_WAZA_ARC", "50")) * 3.141592653589793 / 180.0
#: The shell, as a box in the base frame (m): the black base and the red shell over it.
SHELL_LO = (-0.115, -0.095, -0.012)
SHELL_HI = (0.100, 0.095, 0.075)
#: What a claw can hold besides the shell: segments from a body's origin to the
#: next body's (its child), with a radius. Class 1 legs, class 2 arms.
LEG_SEGMENTS = [(f"{l}_{a}_link", f"{l}_{b}_link") for l in ("LM", "RM", "LR", "RR")
                for a, b in (("thigh", "calf"), ("calf", "foot_tip"))]
ARM_SEGMENTS = [(f"{l}_{a}_link", f"{l}_{b}_link") for l in ("LF", "RF")
                for a, b in (("upper_arm", "forearm"), ("forearm", "palm"))]
SEGMENTS = LEG_SEGMENTS + ARM_SEGMENTS
SEGMENT_CLASS = [1] * len(LEG_SEGMENTS) + [2] * len(ARM_SEGMENTS)
SEGMENT_RADIUS = 0.02
#: Targets: 0 the shell, then the segments. Classes: 0 shell, 1 leg, 2 arm.
TARGET_CLASS = [0] + SEGMENT_CLASS
#: How close a claw tip (its link origin, inside the pincer) must be to take hold,
#: by how wide the claw is open (0 shut to 1 `CLAW_OPEN`): a wide-open pincer
#: closes on something further off, but meets it at the tips of its jaws and
#: holds it less firmly -- a fully open grip keeps `GRIP_WIDE_HOLD` of
#: `GRIP_MAX`. (Until 10/12 every grip took hold at 5 cm and held with 12 N, and
#: how wide a claw opened changed nothing but its look.)
GRIP_RANGE_NARROW, GRIP_RANGE_WIDE = 0.03, 0.07
GRIP_WIDE_HOLD = 0.5
GRIP_K = 300.0         # N/m
GRIP_C = 6.0           # N s/m
GRIP_MAX = 12.0        # N
GRIP_BREAK = 0.07      # m
#: A strike lands when the striking claw tip comes this close to the opponent's
#: shell during the strike's drive.
HIT_RANGE = 0.035
#: A harite is an open-claw slap, as open as the claw was when it began (at least
#: `HARITE_MIN_OPEN`): the wider, the wider its sweep -- it lands from up to
#: `HARITE_REACH` further -- and the longer before the next strike, the claw
#: having to shut again first (up to `HARITE_RECLOSE_S` more cool-down).
HARITE_MIN_OPEN = 0.35
HARITE_REACH = 0.025
HARITE_RECLOSE_S = 0.25


def _mirror(arm: tuple) -> tuple:
    j0, j1, j2, j3 = arm
    return (-j0, j1, -j2, -j3)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate v [..., 3] by unit quaternions q [..., 4] (w, x, y, z)."""
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def _quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return _quat_rotate(torch.cat((q[..., :1], -q[..., 1:]), -1), v)


class Waza:
    """Both wrestlers' claws, for every environment."""

    def __init__(self, env: "ManagerBasedRlEnv"):
        self.env = env
        n, dev = env.num_envs, env.device
        self.device = dev
        z = lambda *s: torch.zeros(n, *s, device=dev)
        #: The sides whose policy commands claws (the other side's arms are left alone).
        self.commanded = [False, False]
        #: Per-environment lean, per side (None: the side's `lean_scale` everywhere): a
        #: league opponent whose roster mixes claw wrestlers trained with different
        #: leans, and wrestlers without claws (0: idle) (`league.py`, 10/14 night).
        self.lean_env: list = [None, None]
        # Commands, per side (and claw).
        self.cmd_open = z(2, 2)
        self.cmd_strike = z(2, 2)
        self.cmd_kind = z(2)
        self.cmd_power = z(2)
        self.cmd_stance = z(2)
        # Grips: the target held (-1 none) and the held point in its frame.
        self.target = torch.full((n, 2, 2), -1, dtype=torch.long, device=dev)
        self.anchor = z(2, 2, 3)
        #: How firmly each grip holds: the share of `GRIP_MAX`, from how wide the
        #: claw was open when it closed.
        self.grip_hold = z(2, 2)
        #: How wide each strike's claw is open: 0 for a closed one, from
        #: `HARITE_MIN_OPEN` to 1 for a harite.
        self.strike_open = z(2, 2)
        self.reach_w = z(2, 2)
        #: "Hold" latched: from the claw output passing 0.2 until it falls below
        #: -0.3. Unlatched, exploration noise turned the reach on and off faster
        #: than the arm could get there, and no grip was ever taken (10/11).
        self.reaching = torch.zeros(n, 2, 2, dtype=torch.bool, device=dev)
        self._lean = torch.tensor(STYLE_LEAN, device=dev)
        self.low_w = z(2, 2)
        self.near = z(2, 2)                                       # nearest target's distance
        self.strike_t = torch.full((n, 2, 2), -1.0, device=dev)   # -1: none
        self.strike_T = torch.full((n, 2, 2), 0.25, device=dev)
        self.strike_kind = torch.zeros(n, 2, 2, dtype=torch.long, device=dev)
        self.cool = z(2, 2)
        self.hit_now = torch.zeros(n, 2, dtype=torch.bool, device=dev)      # landed this step
        self.hit_done = torch.zeros(n, 2, 2, dtype=torch.bool, device=dev)  # this strike has landed
        self.started_now = z(2)                                   # strikes begun this step
        self.since_strike = torch.full((n, 2), 1e3, device=dev)   # since one last landed
        self.last_kind = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self.style = torch.zeros(n, 2, dtype=torch.long, device=dev)
        self._step = None
        self._bodies = None
        allowed = [s for s in os.environ.get("JUMPER_TACTICS_RIKISHI", ",".join(STYLES)).split(",") if s]
        self.allowed = torch.tensor([STYLES.index(s) for s in allowed], device=dev)
        self.box_lo = torch.tensor(SHELL_LO, device=dev)
        self.box_hi = torch.tensor(SHELL_HI, device=dev)
        self.target_class = torch.tensor(TARGET_CLASS, device=dev)
        self.reach_pose = torch.tensor([REACH, _mirror(REACH)], device=dev)            # [claw, 4]
        self.reach_low = torch.tensor([REACH_LOW, _mirror(REACH_LOW)], device=dev)
        self.strike_poses = torch.tensor(
            [[[STRIKE_POSES[k][i] for i in (0, 1)], [_mirror(STRIKE_POSES[k][i]) for i in (0, 1)]]
             for k in KINDS], device=dev)                                             # [kind, claw, 2, 4]

    @property
    def grip(self) -> torch.Tensor:
        """[N, side, claw]: holding something."""
        return self.target >= 0

    def grip_class(self) -> torch.Tensor:
        """[N, side, claw]: what is held, 0 shell 1 leg 2 arm, -1 nothing."""
        return torch.where(self.grip, self.target_class[self.target.clamp(min=0)], -1)

    # -- inputs --------------------------------------------------------------
    def lean_scale(self, side: int) -> float:
        """How much of `STYLE_LEAN` this side's claw outputs are read on top of:
        `JUMPER_WAZA_LEAN` for the learner (east), `JUMPER_SUMO_OPP_WAZA_LEAN` for
        a tactician opponent (1 by default; `JUMPER_WAZA_NO_LEAN=1`: 0 for both).
        The lean was a crutch for finding grips at all; a wrestler that has found
        them uses its claws better on its own: 鋏ノ里 (sogo_waza4) against 回ノ里
        won 18-28 % of its bouts with the lean, 38 % without, 50 % with its claws
        idle (tools/eval/waza_styles.sh, 10/12). In training the learner's lean
        can be taken away over `JUMPER_WAZA_LEAN_DECAY_STEPS` environment steps."""
        if os.environ.get("JUMPER_WAZA_NO_LEAN") == "1":
            return 0.0
        name = "JUMPER_WAZA_LEAN" if side == 0 else "JUMPER_SUMO_OPP_WAZA_LEAN"
        scale = float(os.environ.get(name, "1"))
        decay = float(os.environ.get("JUMPER_WAZA_LEAN_DECAY_STEPS", "0")) if side == 0 else 0.0
        if decay > 0.0:
            scale *= max(0.0, 1.0 - float(self.env.common_step_counter) / decay)
        return scale

    def command(self, side: int, a: torch.Tensor) -> None:
        """A side's seven claw outputs ([N, 7], clipped to [-1, 1])."""
        self.commanded[side] = True
        demo = os.environ.get("JUMPER_WAZA_DEMO")
        if os.environ.get("JUMPER_WAZA_ZERO") == "1":
            a = torch.zeros_like(a)          # diagnostic: claws present but never used
        if demo and side == 0:
            a = self._demo(demo, a)
        else:
            # Each kind of wrestler leans to its own technique: its outputs are
            # read on top of `STYLE_LEAN`, so a yotsu wrestler reaches for the
            # shell whenever the opponent is in reach unless its policy says not
            # to, an oshi wrestler strikes, a waza wrestler goes low for the legs
            # and slaps down. The policy learns when not to.
            scale = self.lean_scale(side)
            per_env = self.lean_env[side]
            if per_env is not None:
                a = (a + per_env[:, None] * self._lean[self.style[:, side]]).clamp(-1.0, 1.0)
            elif scale > 0.0:
                a = (a + scale * self._lean[self.style[:, side]]).clamp(-1.0, 1.0)
            k = min(1.0, self.env.step_dt / WAZA_SMOOTHING_S)
            prev = torch.cat((self.cmd_open[:, side], self.cmd_strike[:, side], self.cmd_kind[:, side, None],
                              self.cmd_power[:, side, None], self.cmd_stance[:, side, None]), -1)
            a = prev + k * (a - prev)
        self.cmd_open[:, side] = a[:, 0:2]
        self.cmd_strike[:, side] = a[:, 2:4]
        self.cmd_kind[:, side] = a[:, 4]
        self.cmd_power[:, side] = a[:, 5]
        self.cmd_stance[:, side] = a[:, 6]

    def _demo(self, name: str, a: torch.Tensor) -> torch.Tensor:
        """A scripted east side, for looking at the mechanics before any policy
        has learned them (`JUMPER_WAZA_DEMO`): "grip" reaches for the shell with
        both claws from the tachiai and closes them 0.6 s later; "legs" does the
        same reaching low; "strikes" throws one strike every 0.6 s, alternating
        arms, through tsuki, harite, hataki, kachiage."""
        from ..mdp import bout

        st = bout(self.env)
        t = (self.env.common_step_counter - st.start_step).float() * self.env.step_dt
        t = torch.where(st.started, t, torch.zeros_like(t))
        out = torch.zeros_like(a)
        if name in ("grip", "legs"):
            # "Hold" from the tachiai on: each claw takes hold by itself once in reach.
            out[:, 0:2] = torch.where((t > 0.15)[:, None], 1.0, 0.0)
            out[:, 4] = -1.0 if name == "legs" else 0.0
            out[:, 6] = 1.0
        elif name == "strikes":
            slot = (t / 0.6).floor().long()
            fire = (t - slot * 0.6) < 0.05
            arm = slot % 2
            kind = (slot // 2) % 4                     # tsuki, harite, hataki, kachiage
            out[:, 2] = torch.where(fire & (arm == 0), 1.0, -1.0)
            out[:, 3] = torch.where(fire & (arm == 1), 1.0, -1.0)
            out[:, 4] = torch.tensor([0.0, 0.0, -1.0, 1.0], device=a.device)[kind]
            harite = (kind == 1) & fire
            out[:, 0] = torch.where(harite & (arm == 0), 1.0, -1.0)
            out[:, 1] = torch.where(harite & (arm == 1), 1.0, -1.0)
            out[:, 5] = 0.5
        return out

    def reset(self, env_ids: torch.Tensor) -> None:
        self.target[env_ids] = -1
        self.reaching[env_ids] = False
        self.hit_done[env_ids] = False
        for t in (self.anchor, self.reach_w, self.low_w, self.cool, self.cmd_open,
                  self.cmd_strike, self.cmd_kind, self.cmd_power, self.cmd_stance,
                  self.grip_hold, self.strike_open):
            t[env_ids] = 0.0
        self.strike_t[env_ids] = -1.0
        self.since_strike[env_ids] = 1e3
        k = torch.randint(len(self.allowed), (len(env_ids), 2), device=self.device)
        self.style[env_ids] = self.allowed[k]

    # -- the bodies ----------------------------------------------------------
    def _ids(self):
        """Per side: base, claw tips, and each target's body (base for the shell,
        then the segments' first bodies) and segment end bodies."""
        if self._bodies is None:
            ids = []
            for side in SIDES:
                ent = self.env.scene[side]
                one = lambda name: ent.find_bodies(name)[0][0]
                base = one("base_link")
                ids.append(dict(base=base, tips=[one(f"{c}F_finger_tip_link") for c in "LR"],
                                seg_a=[one(a) for a, _ in SEGMENTS], seg_b=[one(b) for _, b in SEGMENTS],
                                targets=[base] + [one(a) for a, _ in SEGMENTS]))
            self._bodies = ids
        return self._bodies

    def _frames(self):
        out = []
        for s, side in enumerate(SIDES):
            ids = self._ids()[s]
            d = self.env.scene[side].data
            tb = ids["targets"]
            out.append(dict(
                pos=d.body_link_pos_w[:, ids["base"]], quat=d.body_link_quat_w[:, ids["base"]],
                com=d.body_com_pos_w[:, ids["base"]], vel=d.body_com_lin_vel_w[:, ids["base"]],
                tips=d.body_link_pos_w[:, ids["tips"]],
                seg_a=d.body_link_pos_w[:, ids["seg_a"]], seg_b=d.body_link_pos_w[:, ids["seg_b"]],
                t_pos=d.body_link_pos_w[:, tb], t_quat=d.body_link_quat_w[:, tb],
                t_com=d.body_com_pos_w[:, tb], t_vel=d.body_com_lin_vel_w[:, tb]))
        return out

    def _box_local(self, f_other: dict, p: torch.Tensor) -> torch.Tensor:
        """Points [N, k, 3] into the other side's base frame."""
        q = f_other["quat"][:, None].expand(-1, p.shape[1], -1)
        return _quat_rotate_inv(q, p - f_other["pos"][:, None])

    def _nearest(self, me: dict, other: dict):
        """For each of my claw tips: distance to each of the other's targets
        [N, 2, T] and the nearest point on each, world [N, 2, T, 3]."""
        tips = me["tips"]                                                     # [N, 2, 3]
        local = self._box_local(other, tips)
        on_box = local.clamp(self.box_lo, self.box_hi)
        box_w = other["pos"][:, None] + _quat_rotate(other["quat"][:, None].expand(-1, 2, -1), on_box)
        d_box = torch.linalg.norm(tips - box_w, dim=-1)                       # [N, 2]
        a, b = other["seg_a"][:, None], other["seg_b"][:, None]               # [N, 1, S, 3]
        ab = b - a
        x = tips[:, :, None]                                                  # [N, 2, 1, 3]
        u = (((x - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0.0, 1.0)
        p = a + u[..., None] * ab                                             # [N, 2, S, 3]
        d_seg = (torch.linalg.norm(x - p, dim=-1) - SEGMENT_RADIUS).clamp(min=0.0)
        dist = torch.cat((d_box[..., None], d_seg), -1)                       # [N, 2, T]
        points = torch.cat((box_w[:, :, None], p), 2)                         # [N, 2, T, 3]
        return dist, points

    def _anchor_world(self, s: int, other: dict) -> torch.Tensor:
        """Side s's held points [N, 2, 3], world."""
        t = self.target[:, s].clamp(min=0)                                    # [N, 2]
        pos = torch.gather(other["t_pos"], 1, t[..., None].expand(-1, -1, 3))
        quat = torch.gather(other["t_quat"], 1, t[..., None].expand(-1, -1, 4))
        return pos + _quat_rotate(quat, self.anchor[:, s])

    # -- the step ------------------------------------------------------------
    def update(self) -> None:
        env = self.env
        step = int(env.common_step_counter)
        if step == self._step:
            return
        self._step = step
        from ..mdp import bout

        dt = env.step_dt
        state = bout(env)
        r = getattr(env, "sumo_referee", None)
        live = state.started & ~(r.decided if r is not None else torch.zeros_like(state.started))
        frames = self._frames()
        sep = torch.linalg.norm(frames[0]["pos"][:, :2] - frames[1]["pos"][:, :2], dim=-1)
        near = sep < WAZA_RANGE
        self.hit_now[:] = False
        self.started_now[:] = 0.0
        self.since_strike += dt
        for s in (0, 1):
            me, other = frames[s], frames[1 - s]
            # In reach: near, and in front of this side (`WAZA_ARC`). A latched
            # reach ends when the opponent leaves the front, and the claw comes
            # back to the clay.
            rel = _quat_rotate_inv(me["quat"], other["pos"] - me["pos"])
            close = (near & (torch.atan2(rel[:, 1], rel[:, 0]).abs() < WAZA_ARC))[:, None]
            dist, points = self._nearest(me, other)
            best_d, best = dist.min(-1)                                       # [N, 2]
            self.near[:, s] = best_d
            opening = (self.cmd_open[:, s] > 0.2) & close
            # One claw off the clay at a time: the other stays a front foot. With
            # both claws up a Jumper pitched onto its face and lost "down" (the
            # gated test of 10/11: most of its early losses).
            off = (self.reach_w[:, s] > 0.05) | (self.target[:, s] >= 0) | (self.strike_t[:, s] >= 0)
            first = self.cmd_open[:, s, 0] >= self.cmd_open[:, s, 1]
            pick = torch.stack((first, ~first), -1)
            opening = opening & ~off.flip(-1) & (pick | ~opening.flip(-1))
            # Latched: a reach once begun goes on until told to stop (or the
            # opponent is out of range, or the bout is over).
            stop = (self.cmd_open[:, s] < -0.3) | ~close | ~live[:, None]
            self.reaching[:, s] = (self.reaching[:, s] | opening) & ~stop
            opening = self.reaching[:, s]
            held = self.target[:, s] >= 0
            # How wide each claw is open (0 to 1), as commanded.
            wide = self.open_amount(s)
            # Grips: taken by a reaching claw the moment its tip comes within
            # reach of a target (it closes itself; the wider open, the further
            # off -- and the less firmly it then holds); let go when told (below
            # -0.3); cut when stretched too far; none outside the bout.
            grip_range = GRIP_RANGE_NARROW + wide * (GRIP_RANGE_WIDE - GRIP_RANGE_NARROW)
            take = live[:, None] & ~held & opening & (self.reach_w[:, s] > 0.5) & (best_d < grip_range)
            self.grip_hold[:, s] = torch.where(take, 1.0 - (1.0 - GRIP_WIDE_HOLD) * wide, self.grip_hold[:, s])
            release = self.cmd_open[:, s] < -0.3
            # The claw holds where it closed: its tip, fixed in the held body's
            # frame. (Held at the nearest point on the target, 2-5 cm off, every
            # grip began stretched nearly to breaking.)
            tpos = torch.gather(other["t_pos"], 1, best[..., None].expand(-1, -1, 3))
            tquat = torch.gather(other["t_quat"], 1, best[..., None].expand(-1, -1, 4))
            local = _quat_rotate_inv(tquat, me["tips"] - tpos)
            self.anchor[:, s] = torch.where(take[..., None], local, self.anchor[:, s])
            self.target[:, s] = torch.where(take, best, self.target[:, s])
            stretch = torch.linalg.norm(self._anchor_world(s, other) - me["tips"], dim=-1)
            keep = held & ~release & (stretch < GRIP_BREAK) & live[:, None]
            self.target[:, s] = torch.where(keep | take, self.target[:, s], -1)
            # Strikes: begun by a free arm, run their course, then cool down.
            striking = self.strike_t[:, s] >= 0
            begin = live[:, None] & close & (self.cmd_strike[:, s] > 0.5) & ~striking & ~self.grip[:, s] \
                & (self.cool[:, s] <= 0)
            firstb = self.cmd_strike[:, s, 0] >= self.cmd_strike[:, s, 1]
            pickb = torch.stack((firstb, ~firstb), -1)
            begin = begin & ~off.flip(-1) & ~opening.flip(-1) & (pickb | ~begin.flip(-1))
            kind = torch.where(self.cmd_kind[:, s] < -0.33, 2, torch.where(self.cmd_kind[:, s] > 0.33, 3, 0))
            kind = kind[:, None].expand(-1, 2)
            kind = torch.where((kind == 0) & opening, torch.ones_like(kind), kind)   # open: harite
            T = STRIKE_SLOW_S + (STRIKE_FAST_S - STRIKE_SLOW_S) * (self.cmd_power[:, s] + 1) / 2
            self.strike_kind[:, s] = torch.where(begin, kind, self.strike_kind[:, s])
            harite_open = torch.where(kind == 1, wide.clamp(min=HARITE_MIN_OPEN), torch.zeros_like(wide))
            self.strike_open[:, s] = torch.where(begin, harite_open, self.strike_open[:, s])
            self.strike_T[:, s] = torch.where(begin, T[:, None].expand(-1, 2), self.strike_T[:, s])
            self.strike_t[:, s] = torch.where(begin, torch.zeros_like(self.strike_t[:, s]),
                                              torch.where(striking, self.strike_t[:, s] + dt,
                                                          self.strike_t[:, s]))
            self.hit_done[:, s] &= ~begin
            self.started_now[:, s] = begin.float().sum(-1)
            u = self.strike_t[:, s] / self.strike_T[:, s]
            driving = (u >= 0.3) & (u <= 0.85)
            hit = driving & (dist[..., 0] < HIT_RANGE + HARITE_REACH * self.strike_open[:, s]) \
                & ~self.hit_done[:, s]
            self.hit_done[:, s] |= hit
            landed = hit.any(-1)
            self.hit_now[:, s] = landed
            self.last_kind[:, s] = torch.where(
                landed, torch.where(hit[:, 0], self.strike_kind[:, s, 0], self.strike_kind[:, s, 1]),
                self.last_kind[:, s])
            self.since_strike[:, s] = torch.where(landed, torch.zeros_like(self.since_strike[:, s]),
                                                  self.since_strike[:, s])
            done = (self.strike_t[:, s] >= self.strike_T[:, s]) | ~live[:, None]
            self.cool[:, s] = torch.where(done & (self.strike_t[:, s] >= 0),
                                          STRIKE_COOL_S + HARITE_RECLOSE_S * self.strike_open[:, s],
                                          (self.cool[:, s] - dt).clamp(min=0.0))
            self.strike_t[:, s] = torch.where(done, torch.full_like(self.strike_t[:, s], -1.0),
                                              self.strike_t[:, s])
            # Reaching: an open claw (or a holding one), free of strikes, eased in
            # over 0.1 s; low for the legs when `kind` is below 0 (held low when a
            # leg is held).
            want = (opening | self.grip[:, s]) & live[:, None] & (self.strike_t[:, s] < 0)
            self.reach_w[:, s] += (want.float() - self.reach_w[:, s]) * min(1.0, dt / 0.1)
            low = torch.where(self.grip[:, s], self.grip_class()[:, s] == 1,
                              (self.cmd_kind[:, s] < 0)[:, None].expand(-1, 2))
            self.low_w[:, s] += (low.float() - self.low_w[:, s]) * min(1.0, dt / 0.1)
            if os.environ.get("JUMPER_WAZA_DEBUG") and s == 0 and step % 25 == 0:
                print(f"\n[waza] tick live={bool(live[0])} open={self.cmd_open[0, s].tolist()} "
                      f"reach_w={[round(v, 2) for v in self.reach_w[0, s].tolist()]} "
                      f"low_w={[round(v, 2) for v in self.low_w[0, s].tolist()]} "
                      f"near={[round(v, 3) for v in best_d[0].tolist()]} best={best[0].tolist()}", flush=True)
            if os.environ.get("JUMPER_WAZA_DEBUG") and (bool(take[0].any()) or bool(begin[0].any())
                                                        or bool(landed[0]) or bool((held[0] & ~keep[0]).any())):
                t = float((env.common_step_counter - state.start_step[0]) * dt)
                print(f"\n[waza] {SIDES[s]} t={t:5.2f} take={take[0].tolist()} held={self.grip_class()[0, s].tolist()} "
                      f"target={self.target[0, s].tolist()} begin={begin[0].tolist()} "
                      f"kind={self.strike_kind[0, s].tolist()} hit={bool(landed[0])} "
                      f"stretch={[round(v, 3) for v in stretch[0].tolist()]} "
                      f"near={[round(v, 3) for v in best_d[0].tolist()]}", flush=True)
        self._apply_grips(frames)

    def _apply_grips(self, frames) -> None:
        """Equal and opposite spring forces: on the held body at the held point, and
        on the holder's base at its claw tip."""
        n = self.env.num_envs
        nt = len(TARGET_CLASS)
        force = [torch.zeros(n, nt, 3, device=self.device) for _ in SIDES]     # per target body
        torque = [torch.zeros(n, nt, 3, device=self.device) for _ in SIDES]
        for s in (0, 1):
            if not bool(self.grip[:, s].any()):
                continue
            me, other = frames[s], frames[1 - s]
            anchor_w = self._anchor_world(s, other)                           # [N, 2, 3]
            t = self.target[:, s].clamp(min=0)
            t_vel = torch.gather(other["t_vel"], 1, t[..., None].expand(-1, -1, 3))
            t_com = torch.gather(other["t_com"], 1, t[..., None].expand(-1, -1, 3))
            f = GRIP_K * (anchor_w - me["tips"]) + GRIP_C * (t_vel - me["vel"][:, None])
            norm = torch.linalg.norm(f, dim=-1, keepdim=True)
            cap = GRIP_MAX * self.grip_hold[:, s, :, None]                    # [N, 2, 1]
            f = f * (cap / norm.clamp(min=1e-6)).clamp(max=1.0)
            f = torch.where(self.grip[:, s][..., None], f, torch.zeros_like(f))   # [N, 2, 3] on the holder
            # The holder: at its claw tips, on its base (target slot 0 is its base).
            force[s][:, 0] += f.sum(1)
            torque[s][:, 0] += torch.cross(me["tips"] - me["com"][:, None], f, dim=-1).sum(1)
            # The held: at the held points, on the held bodies.
            tq = torch.cross(anchor_w - t_com, -f, dim=-1)
            force[1 - s].scatter_add_(1, t[..., None].expand(-1, -1, 3), -f)
            torque[1 - s].scatter_add_(1, t[..., None].expand(-1, -1, 3), tq)
        for s, side in enumerate(SIDES):
            self.env.scene[side].write_external_wrench_to_sim(
                force[s], torque[s], body_ids=self._ids()[s]["targets"])

    # -- the arms ------------------------------------------------------------
    def open_amount(self, side: int) -> torch.Tensor:
        """[N, claw]: how wide each claw of this side is commanded open, 0 (shut;
        a claw output of 0.2 and below) to 1 (`CLAW_OPEN`, an output of 1)."""
        return ((self.cmd_open[:, side] - 0.2) / 0.8).clamp(0.0, 1.0)

    def stance_depth(self, entity: str) -> torch.Tensor:
        """0 (the usual shikiri stance) to 1 (`arms.STANCE_LOW`), as chosen."""
        return self.cmd_stance[:, SIDES.index(entity)].clamp(0.0, 1.0)

    def overlay(self, term, entity: str) -> None:
        """Lay this side's claws over its action term's joint targets (after the
        shikiri's): reach / hold / strike poses and the claw openings (J4)."""
        from ..mdp import bout

        side = SIDES.index(entity)
        self.update()
        if not self.commanded[side]:
            return
        state = bout(self.env)
        names = [f"{c}F_J{k}_joint" for c in "LR" for k in range(4)]
        idx = _joint_index(term, names).view(2, 4)
        p = term._processed_actions
        for c in (0, 1):
            cols = idx[c]
            cur = p[:, cols]
            lw = self.low_w[:, side, c:c + 1]
            reach = self.reach_pose[c] + lw * (self.reach_low[c] - self.reach_pose[c])
            w = self.reach_w[:, side, c:c + 1]
            cur = cur + w * (reach - cur)
            # A strike: to the wind-up, through to the finish, and back.
            t = self.strike_t[:, side, c]
            active = t >= 0
            if bool(active.any()):
                u = (t / self.strike_T[:, side, c]).clamp(0.0, 1.0)[:, None]
                poses = self.strike_poses[self.strike_kind[:, side, c], c]          # [N, 2, 4]
                drive = ((u - 0.3) / 0.45).clamp(0.0, 1.0)
                drive = drive * drive * (3 - 2 * drive)
                target = poses[:, 0] + drive * (poses[:, 1] - poses[:, 0])
                ws = torch.where(u < 0.3, u / 0.3, torch.where(u > 0.85, (1 - u) / 0.15, torch.ones_like(u)))
                ws = torch.where(active[:, None], ws, torch.zeros_like(ws))
                cur = cur + ws * (target - cur)
            p[:, cols] = cur
        # The claws: open as commanded (a harite as open as it began), closed
        # when holding.
        opening = self.open_amount(side)
        harite = (self.strike_t[:, side] >= 0) & (self.strike_kind[:, side] == 1)
        opening = torch.where(harite, self.strike_open[:, side], opening)
        opening = torch.where(self.grip[:, side], torch.zeros_like(opening), opening)
        live = state.started.float()[:, None]
        j4 = opening * CLAW_OPEN * live * torch.tensor([-1.0, 1.0], device=self.device)
        term._entity.set_joint_position_target(j4, joint_ids=_j4_ids(term))


def _joint_index(term, names: list[str]) -> torch.Tensor:
    cache = term.__dict__.setdefault("_waza_idx", {})
    key = tuple(names)
    if key not in cache:
        targets = list(term.target_names)
        cache[key] = torch.tensor([next(i for i, n in enumerate(targets) if n.endswith(j)) for j in names],
                                  device=term.device)
    return cache[key]


def _j4_ids(term):
    ids = term.__dict__.get("_waza_j4")
    if ids is None:
        ids = term._entity.find_joints(["LF_J4_joint", "RF_J4_joint"], preserve_order=True)[0]
        term.__dict__["_waza_j4"] = ids
    return ids


def waza(env: "ManagerBasedRlEnv") -> Waza | None:
    """The environment's claws, made on first use when either side has them
    (`JUMPER_TACTICS_WAZA=1` for the learner, `JUMPER_SUMO_OPP_WAZA=1` for a
    tactician opponent); None otherwise."""
    w = getattr(env, "sumo_waza", None)
    if w is None and "1" in (os.environ.get("JUMPER_TACTICS_WAZA", ""),
                             os.environ.get("JUMPER_SUMO_OPP_WAZA", "")):
        w = Waza(env)
        env.sumo_waza = w
    return w


# ── Observations ─────────────────────────────────────────────────────────────


def claws_state(env: "ManagerBasedRlEnv", me: str = "robot") -> torch.Tensor:
    """What a waza tactician knows of the claws (22): its own claw tips in the
    opponent's base frame (6, clipped to +-0.5 m); what each claw holds (one-hot
    shell / leg / arm, 6); its strikes' progress (2); whether the opponent holds
    it, per opponent claw (2); how near each claw tip is to something it could
    hold (2, m, clipped to 0.3); and the kind of wrestler it is being (`STYLES`,
    one-hot, 4)."""
    w = waza(env)
    if w is None:
        return torch.zeros(env.num_envs, CLAWS_OBS_DIM, device=env.device)
    s = SIDES.index(me)
    frames = w._frames()
    tips = w._box_local(frames[1 - s], frames[s]["tips"]).clamp(-0.5, 0.5).reshape(env.num_envs, 6)
    held = torch.nn.functional.one_hot(w.grip_class()[:, s].clamp(min=0), 3).float() \
        * w.grip[:, s, :, None].float()
    strike = torch.where(w.strike_t[:, s] >= 0, w.strike_t[:, s] / w.strike_T[:, s],
                         torch.zeros_like(w.strike_t[:, s]))
    style = torch.nn.functional.one_hot(w.style[:, s], len(STYLES)).float()
    return torch.cat((tips, held.reshape(env.num_envs, 6), strike, w.grip[:, 1 - s].float(),
                      w.near[:, s].clamp(max=0.3), style), -1)


# ── Rewards (the robot, east) ────────────────────────────────────────────────

#: Per kind (rows: oshi, yotsu, nage, waza) and what is held (shell, leg, arm):
#: a hold kept, per second and claw.
GRIP_PAY = ((-0.3, -0.3, -0.3), (2.5, 0.5, 0.5), (1.5, 0.3, 0.6), (0.0, 2.0, 2.0))
#: A strike that lands, once; a hataki that lands, once; the opponent turned
#: while held, per rad/s.
HIT_PAY = (2.0, 0.2, 0.2, 0.0)
HATAKI_PAY = (1.0, 0.0, 0.0, 3.0)
SPIN_PAY = (0.0, 0.0, 1.0, 0.0)
#: The kimarite each kind is paid a bonus for winning by.
STYLE_KIMARITE = {
    "oshi": ("oshidashi", "oshitaoshi", "tsukidashi", "tsukitaoshi"),
    "yotsu": ("yorikiri", "yoritaoshi"),
    "nage": ("uwatenage", "shitatenage"),
    "waza": ("hatakikomi", "hikiotoshi", "ashitori", "tottari", "okuridashi", "tsukiotoshi"),
}


def _pay(w: Waza, table: tuple) -> torch.Tensor:
    return torch.tensor(table, device=w.device)[w.style[:, 0]]


def grip_held(env: "ManagerBasedRlEnv") -> torch.Tensor:
    w = waza(env)
    if w is None:
        return torch.zeros(env.num_envs, device=env.device)
    table = torch.tensor(GRIP_PAY, device=w.device)[w.style[:, 0]]           # [N, 3]
    cls = w.grip_class()[:, 0]                                                # [N, 2]
    pay = torch.gather(table, 1, cls.clamp(min=0))
    return (pay * w.grip[:, 0].float()).sum(-1)


def strike_landed(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """1 (scaled by kind) on the step a strike lands."""
    w = waza(env)
    if w is None:
        return torch.zeros(env.num_envs, device=env.device)
    hataki = w.last_kind[:, 0] == 2
    pay = torch.where(hataki, _pay(w, HATAKI_PAY), _pay(w, HIT_PAY))
    return w.hit_now[:, 0].float() * pay


def strike_begun(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Strikes begun this step (a small cost, so they are not thrown for nothing)."""
    w = waza(env)
    return torch.zeros(env.num_envs, device=env.device) if w is None else w.started_now[:, 0]


def turned_while_held(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """The opponent's yaw rate while the robot holds its shell (rad/s), nage only."""
    w = waza(env)
    if w is None:
        return torch.zeros(env.num_envs, device=env.device)
    spin = env.scene["opponent"].data.root_link_ang_vel_w[:, 2].abs().clamp(max=4.0)
    shell = (w.grip_class()[:, 0] == 0).any(-1)
    return shell.float() * spin * _pay(w, SPIN_PAY)


#: The kimarite only claws win by (`kimarite_hint`): a leg or arm held, a strike,
#: a throw from a hold. Paid, whatever the kind, when `JUMPER_WAZA_CLAW_BONUS` is
#: set (tactics/env_cfg.py): the strongest claw wrestlers of 10/13 won fewer than
#: 1 % of their bouts by them, their claws better used as front feet.
CLAW_KIMARITE = ("ashitori", "tottari", "tsukidashi", "tsukitaoshi", "hatakikomi",
                 "uwatenage", "shitatenage")


def claw_kimarite_won(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """1 on the step the robot wins by one of `CLAW_KIMARITE`."""
    from ..referee import KIMARITE, referee

    r = referee(env, radius)
    r.update()
    names = [n for n, _ in KIMARITE]
    claw = torch.zeros(len(KIMARITE), dtype=torch.bool, device=env.device)
    for k in CLAW_KIMARITE:
        if k in names:
            claw[names.index(k)] = True
    now = r.decided & (r.decided_at == r.time) & (r.winner == 0)
    return (now & claw[r.kimarite]).float()


def style_kimarite_won(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """1 on the step the robot wins by one of its kind's kimarite."""
    from ..referee import KIMARITE, referee

    w = waza(env)
    r = referee(env, radius)
    r.update()
    if w is None:
        return torch.zeros(env.num_envs, device=env.device)
    names = [n for n, _ in KIMARITE]
    table = torch.zeros(len(STYLES), len(KIMARITE), dtype=torch.bool, device=env.device)
    for i, style in enumerate(STYLES):
        for k in STYLE_KIMARITE[style]:
            if k in names:
                table[i, names.index(k)] = True
    now = r.decided & (r.decided_at == r.time) & (r.winner == 0)
    return (now & table[w.style[:, 0], r.kimarite]).float()


# ── The referee's eye ────────────────────────────────────────────────────────


def kimarite_hint(env: "ManagerBasedRlEnv", rows: torch.Tensor, winner: torch.Tensor,
                  loser: torch.Tensor, out: torch.Tensor, fell: torch.Tensor,
                  k: torch.Tensor, K: dict) -> torch.Tensor:
    """The referee's kimarite (`k`), corrected by what the claws did at the
    decision. The winner holding the loser's shell: yorikiri when the loser is
    out; down and turned, a throw (uwatenage when the hold is on the loser's
    flank, shitatenage on its front); down and not turned, yoritaoshi. Holding a
    leg: ashitori. Holding an arm: tottari. Its strike landed within 0.5 s:
    tsukidashi out, tsukitaoshi down, or hatakikomi for a slap down."""
    w = getattr(env, "sumo_waza", None)
    if w is None:
        return k
    cls = w.grip_class()[rows, winner]                                     # [N, 2]
    shell = (cls == 0).any(-1)
    leg = (cls == 1).any(-1)
    arm = (cls == 2).any(-1)
    anchor = w.anchor[rows, winner]                                        # [N, 2, 3]
    flank = ((anchor[..., 1].abs() > 0.06) & (cls == 0)).any(-1)
    spin = torch.stack((env.scene["robot"].data.root_link_ang_vel_w[:, 2],
                        env.scene["opponent"].data.root_link_ang_vel_w[:, 2]), -1)[rows, loser].abs()
    recent = w.since_strike[rows, winner] < 0.5
    kind = w.last_kind[rows, winner]
    thrust = recent & ((kind <= 1) | (kind == 3))
    decided = out | fell
    k = torch.where(thrust & out, K["tsukidashi"], k)
    k = torch.where(thrust & fell, K["tsukitaoshi"], k)
    k = torch.where(recent & (kind == 2) & fell, K["hatakikomi"], k)
    k = torch.where(arm & decided, K["tottari"], k)
    k = torch.where(leg & decided, K["ashitori"], k)
    k = torch.where(shell & out, K["yorikiri"], k)
    k = torch.where(shell & fell, torch.where(
        spin > 1.5, torch.where(flank, K["uwatenage"], K["shitatenage"]), K["yoritaoshi"]), k)
    return k
