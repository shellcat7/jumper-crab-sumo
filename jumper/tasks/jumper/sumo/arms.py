"""Arm poses laid over a policy's joint targets (jumper-challenge-2026).

Two rituals no policy here was trained for, both done by blending the front arms'
joint targets towards a fixed pose after the policy has chosen its own:

- **Shikiri with the hands raised.** Each wrestler stands with both claws held up
  in front of it, a little below its face, then lowers them -- each at its own
  moment, `BoutState.lower_at` -- and the bout begins once both pairs of claws
  are down (`mdp.BoutState`). A Jumper walks on its claws, so with them down from
  the start "hands down" was never something one could see happen. Through the
  whole shikiri the wrestler's joints are held to this ritual, legs included, and
  given back to its policy at the tachiai: a policy trained to push learned to
  charge the moment its claws touched (in training the bout began then), so left
  to itself it crept and false-started whatever it was told (sumo_cont700, 10/10).
- **The gyoji's gunbai.** The gyoji carries it in its right claw (`gyoji.py`),
  held up in front of it, and raises it higher when it calls the winner.

- **Stop at the decision.** Once the gyoji has called the bout (a replay with a
  gyoji, where the ring is not reset the moment it is decided), both wrestlers'
  joints are taken back to the stance over `STOP_S`: a bout ends at "shobu ari",
  and before this they went on pushing for the 1.2 s the call is shown (the user
  saw it, 10/10 night).

Between the tachiai and the decision a wrestler's joints are all its policy's; the
gyoji's right arm (`*F_J0..J3` of RF) is the only joint held for the whole bout.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg

from ..common.constants import GAIT_JOINTS, HOME

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: Both claws raised in front of the body, the claw tips a little below its top:
#: shoulders (J1) up 0.93 rad from HOME's -1.07, forearms folded (J2, J3), arms
#: drawn in a little (J0). Mirrored left to right. In the shikiri stance (body
#: level at 9.3 cm, `STANCE`) the tips are 11.5 cm off the clay, 2 cm below the
#: body's top. (Before the stance kept the body level it pitched forward, and
#: J1 -2.6 was what lifted the tips to 8.6 cm.) The first
#: version (J1 -1.85, forearms as `five_foot`'s stow) left the tips 2.2 cm up --
#: on video the claws looked already down before "te o tsuite" (the user saw it).
RAISED: dict[str, float] = {
    "LF_J0_joint": -0.35, "LF_J1_joint": -2.00, "LF_J2_joint": -1.00, "LF_J3_joint": -1.60,
    "RF_J0_joint": 0.35, "RF_J1_joint": -2.00, "RF_J2_joint": 1.00, "RF_J3_joint": 1.60,
}
if os.environ.get("JUMPER_SUMO_RAISED"):
    # Trying other raised poses: "J0,J1,J2,J3" for the left arm, mirrored to the right.
    _j0, _j1, _j2, _j3 = (float(v) for v in os.environ["JUMPER_SUMO_RAISED"].split(","))
    RAISED = {"LF_J0_joint": _j0, "LF_J1_joint": _j1, "LF_J2_joint": _j2, "LF_J3_joint": _j3,
              "RF_J0_joint": -_j0, "RF_J1_joint": _j1, "RF_J2_joint": -_j2, "RF_J3_joint": -_j3}
GYOJI_HOLD: dict[str, float] = {
    "RF_J0_joint": 0.45, "RF_J1_joint": -1.95, "RF_J2_joint": 0.55, "RF_J3_joint": 1.20,
}
GYOJI_RAISE: dict[str, float] = {
    "RF_J0_joint": 0.45, "RF_J1_joint": -2.45, "RF_J2_joint": 0.55, "RF_J3_joint": 0.80,
}
#: Seconds a wrestler takes to bring its claws down (the moment worth seeing).
LOWER_S = 0.8
#: The shikiri stance under the arms: the robot's standing HOME, claws down -- the
#: pose the arms are lowered into -- except the middle legs, swung forward at the
#: hip. With both claws up a Jumper stands on its middle and rear feet, and at HOME
#: its centre of mass (x +0.003 m with the claws raised) is ahead of the middle
#: feet (x -0.007): it pitched onto its face and its rear legs left the clay (the
#: user saw it, 10/10). Swung 0.5 rad, the middle feet are at x +0.06.
STANCE: dict[str, float] = {
    **{j: HOME[j] for j in GAIT_JOINTS}, "LM_J0_joint": -0.5, "RM_J0_joint": 0.5,
}
#: The same stance crouched 2.5 cm lower, every foot where it was (solved by
#: forward kinematics, 10/10 night): the low shikiri a waza tactician can choose
#: (`tactics/waza.py`, its "stance" output). A low stance is the defensive one in
#: the sources: university wrestlers who crouched low at the tachiai did so to
#: receive the charge, not to hit harder (docs/notes/rikishi-and-waza.md).
STANCE_LOW: dict[str, float] = {
    **STANCE,
    "LM_J1_joint": 1.0356, "LM_J2_joint": -2.1771, "RM_J1_joint": -1.0356, "RM_J2_joint": 2.1771,
    "LR_J1_joint": 1.0534, "LR_J2_joint": -2.2148, "RR_J1_joint": -1.0534, "RR_J2_joint": 2.2148,
    "LF_J1_joint": -1.2158, "LF_J2_joint": -0.2408, "LF_J3_joint": -1.624,
    "RF_J1_joint": -1.2158, "RF_J2_joint": 0.2408, "RF_J3_joint": 1.624,
}


def overlay(term, pose: dict[str, float], weight: torch.Tensor) -> None:
    """Blend `term`'s joint targets for `pose`'s joints towards `pose` by
    `weight` ([N], 0 = the policy's targets, 1 = the pose)."""
    cache = term.__dict__.setdefault("_arm_overlay", {})
    key = tuple(pose)
    if key not in cache:
        names = list(term.target_names)
        idx = [next(i for i, n in enumerate(names) if n.endswith(j)) for j in pose]
        cache[key] = (torch.tensor(idx, device=term.device),
                      torch.tensor(list(pose.values()), device=term.device))
    idx, target = cache[key]
    w = weight.clamp(0.0, 1.0)[:, None]
    p = term._processed_actions
    p[:, idx] = w * target + (1 - w) * p[:, idx]


#: Seconds over which both wrestlers are brought to a stop at the decision.
STOP_S = 0.25


def stop_weight(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """0 until the gyoji has called the bout, rising to 1 over `STOP_S` after it.
    Always 0 without a gyoji: there the decision ends the episode."""
    r = getattr(env, "sumo_referee", None)
    if r is None or not r.officiates:
        return torch.zeros(env.num_envs, device=env.device)
    w = ((r.time - r.decided_at) / STOP_S).clamp(0.0, 1.0)
    return torch.where(r.decided, w, torch.zeros_like(w))


def shikiri_weight(env: "ManagerBasedRlEnv", entity: str) -> torch.Tensor:
    """1 while `entity`'s claws are still up, falling to 0 over `LOWER_S` from its
    own lowering moment; 0 once the bout has begun."""
    from .mdp import bout

    state = bout(env)
    col = 0 if entity == "robot" else 1
    w = (state.lower_at[:, col] + LOWER_S - state.elapsed) / LOWER_S
    return torch.where(state.started, torch.zeros_like(w), w.clamp(0.0, 1.0))


#: The claws' ceremony before a replay's shikiri (`JUMPER_SUMO_CEREMONY=seconds`, 10/20; 0,
#: the default, everywhere else -- no policy was trained with it). Both wrestlers, claws
#: raised as for the shikiri: two claps of the pincers (a crab's kashiwade), then both claws
#: thrown up and out in a V with the pincers wide open, held, and brought back (chiri-chozu,
#: the open hands that show nothing is hidden), then the shikiri proper. In seconds from
#: the start of the shikiri:
CLAP_AT, CLAP_S = 0.8, 1.2          # two snaps, each 0.6 s
SPREAD_AT, SPREAD_S = 2.2, 2.0      # up over 0.6 s, held 0.8 s, back over 0.6 s
#: The V, as offsets from `RAISED` for the left arm (the right mirrored: J0, J2, J3 negated).
#: Swinging J0 outwards (dc491db's spread, +0.55) drove the arm through the middle leg swung
#: forward for the stance (2.3 cm deep, 10/21); this pose was searched for by forward
#: kinematics: the claws 1.6 cm clear of every middle and rear leg geom, and up and out.
SPREAD_LF = {"LF_J0_joint": -0.3, "LF_J1_joint": -1.6, "LF_J2_joint": 0.8, "LF_J3_joint": 0.8}
SNAP = 1.0                          # pincers opened for a clap, rad (LF negative, RF positive)
SPREAD_OPEN = 1.2                   # and held open in the V


def ceremony_s() -> float:
    v = float(os.environ.get("JUMPER_SUMO_CEREMONY", "0") or 0.0)
    return max(v, SPREAD_AT + SPREAD_S + 0.4) if v > 0 else 0.0


def _spread(t: torch.Tensor) -> torch.Tensor:
    """0 -> 1 -> 0 over the spread (smoothstep up and down, held between)."""
    v = ((t - SPREAD_AT) / SPREAD_S).clamp(0.0, 1.0)
    ramp = torch.clamp(torch.minimum(v / 0.3, (1.0 - v) / 0.3), 0.0, 1.0)
    return ramp * ramp * (3 - 2 * ramp)


def _overlay_values(term, names: tuple[str, ...], values: torch.Tensor, weight: torch.Tensor) -> None:
    """As `overlay`, but per-environment target values [N, len(names)]."""
    cache = term.__dict__.setdefault("_arm_overlay_v", {})
    if names not in cache:
        tn = list(term.target_names)
        cache[names] = torch.tensor([next(i for i, n in enumerate(tn) if n.endswith(j)) for j in names],
                                    device=term.device)
    idx = cache[names]
    w = weight.clamp(0.0, 1.0)[:, None]
    p = term._processed_actions
    p[:, idx] = w * values + (1 - w) * p[:, idx]


def ceremony_offsets(term, entity: str) -> None:
    """The ceremony's V of the claws, laid over the raised claws while the shikiri's clock
    is inside the ceremony."""
    from .mdp import bout

    state = bout(term._env)
    c = getattr(state, "ceremony", 0.0)
    if c <= 0 or entity not in ("robot", "opponent"):
        return
    t = state.elapsed
    live = (~state.started) & (t < c)
    e = _spread(t)
    names = tuple(SPREAD_LF) + tuple(n.replace("LF", "RF") for n in SPREAD_LF)
    off = list(SPREAD_LF.values()) + [v if n == "LF_J1_joint" else -v for n, v in SPREAD_LF.items()]
    base = torch.tensor([RAISED[n] for n in names], device=t.device)
    off = torch.tensor(off, device=t.device)
    _overlay_values(term, names, base + e[:, None] * off, (live & (e > 0)).float())


def ceremony_claps(term, entity: str) -> None:
    """The ceremony's two pincer claps, and the pincers held open in the V. The pincers
    (J4) are not in a wrestler's action (10/21: the first replay stopped on it), so their
    targets are set on the entity, after the claws' own (`Waza.overlay` holds them shut
    until the tachiai)."""
    from .mdp import bout

    state = bout(term._env)
    c = getattr(state, "ceremony", 0.0)
    if c <= 0 or entity not in ("robot", "opponent"):
        return
    t = state.elapsed
    live = (~state.started) & (t < c)
    u = ((t - CLAP_AT) / CLAP_S).clamp(0.0, 1.0)
    snap = torch.where((t > CLAP_AT) & (t < CLAP_AT + CLAP_S), 0.5 - 0.5 * torch.cos(4 * math.pi * u),
                       torch.zeros_like(t))
    snap = torch.maximum(SNAP * snap, SPREAD_OPEN * _spread(t))
    ids = term.__dict__.get("_ceremony_j4")
    if ids is None:
        ids = term._entity.find_joints(["LF_J4_joint", "RF_J4_joint"], preserve_order=True)[0]
        term.__dict__["_ceremony_j4"] = ids
    cur = term._entity.data.joint_pos_target[:, ids]
    vals = torch.stack((-snap, snap), -1)
    term._entity.set_joint_position_target(torch.where(live[:, None], vals, cur), joint_ids=ids)


def shikiri_arms(term, entity: str) -> None:
    """The raised-hands shikiri for a wrestler's action term (no-op for others):
    held at `STANCE` until the tachiai, the claws at `RAISED` until it lowers them;
    held at `STANCE` again once the bout is called (`stop_weight`). With claws
    (`tactics/waza.py`): the crouch as deep as the wrestler chose, and after the
    tachiai its reaches, grips, strikes and claw openings."""
    if entity in ("robot", "opponent"):
        from .mdp import bout
        from .tactics.waza import waza

        env = term._env
        waiting = (~bout(env).started).float()
        overlay(term, STANCE, torch.maximum(waiting, stop_weight(env)))
        w = waza(env)
        if w is not None:
            overlay(term, STANCE_LOW, waiting * w.stance_depth(entity))
        overlay(term, RAISED, shikiri_weight(env, entity))
        ceremony_offsets(term, entity)
        if w is not None:
            w.overlay(term, entity)
        ceremony_claps(term, entity)


@dataclass(kw_only=True)
class RitualJointPositionActionCfg(JointPositionActionCfg):
    """The learner's own joint action in `jumper.sumo`, with the shikiri arms."""

    def build(self, env: "ManagerBasedRlEnv") -> "RitualJointPositionAction":
        return RitualJointPositionAction(self, env)


class RitualJointPositionAction(JointPositionAction):
    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        shikiri_arms(self, self.cfg.entity_name)
