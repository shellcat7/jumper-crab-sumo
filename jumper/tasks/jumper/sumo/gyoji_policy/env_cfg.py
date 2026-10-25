"""Environment config for jumper.sumo.gyoji_policy (jumper-challenge-2026).

`jumper.sumo` with the gyoji always in the ring, and the gyoji the learner:

- **Both wrestlers frozen**, each drawn every bout from a roster (`JUMPER_SUMO_LEAGUE`,
  the round robin's format; `tactics/league.py`) -- walkers and tacticians, the
  pusher, the circler, the reader -- so the gyoji meets every kind of bout. The
  robot's own joint action is gone; the east side is a league as the west is.
- **The gyoji's action** is `mdp.LearnedGyojiAction`: four numbers to its frozen
  walking policy, the gunbai held as the scripted gyoji holds it.
- **What it sees** (`mdp.gyoji_policy_terms`, 44): its own motion, each wrestler
  in its frame, the pair as a line, the ring (and which way the front is), the
  bout's phase and who won, its last choice. The critic also knows whether it is
  touching a wrestler.
- **What it is paid for** (`mdp.py`'s module docstring has the sources): a side
  view of the pair from the watching distance, facing them, from the far side of
  the front when it can; never touched, never in reach, not out on the straw;
  moving only as much as it must and without rocking; crouched to see the feet
  when the bout reaches the straw; standing, facing the winner, at the decision.
  Falling or leaving the mound ends the episode with a penalty.
- An episode is a bout: the shikiri (the gyoji stands), the bout, the decision and
  the 2 s the call is shown (`referee.bout_over`), or 20 s.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path

from mjlab.envs.mdp.rewards import action_rate_l2
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ..env_cfg import _NEUTRAL_HEIGHT, RING_RADIUS
from ..env_cfg import env_cfg as sumo_env_cfg
from ..gyoji import GYOJI
from ..opponent import OPPONENT, HeldCommandCfg
from .mdp import (
    LearnedGyojiActionCfg, back_to_front, beyond_straw, bobbing, crouch_at_the_straw, facing,
    gyoji_ended, gyoji_fell, gyoji_off, gyoji_policy_terms, gyoji_touched, hidden_from_front, moving,
    saw_decision, standing_after, too_close, touching, turning, watch_distance, watch_side,
)

#: Rewards, per second of the bout (the fall's once). Calibrated on rollouts (10/14,
#: 8 rings, the roster): with bobbing at -0.2 a gyoji that never moved scored more
#: than the scripted one (0.37 against 0.23 an episode-second), the scripted one
#: paying 0.15 for its rocking walk -- walking itself was the cost. At -0.08 the
#: two are even, and only a gyoji that moves to the right place beats both.
WEIGHTS = {
    "watch_side": 1.5, "watch_distance": 0.5, "facing": 0.5, "back_to_front": 0.25,
    "too_close": -2.0, "touching": -4.0, "beyond_straw": -1.0,
    "moving": -1.5, "turning": -0.1, "bobbing": -0.08, "action_rate": -0.02,
    "crouch_at_the_straw": 0.3, "standing_after": 0.5,
    # 10/15: hidden from the front's camera behind a wrestler (gyoji8); 0 keeps the
    # earlier gyoji's rewards (JUMPER_GYOJI_WEIGHTS=hidden=-0.5 turns it on).
    "hidden": 0.0,
    # 10/20: what one decision pays when the gyoji saw it clearly (mdp.saw_decision); 0 keeps
    # the earlier gyoji's rewards (JUMPER_GYOJI_WEIGHTS=saw_decision=6 for gyoji13).
    "saw_decision": 0.0,
}
FALL_PENALTY = 20.0


def env_cfg(asset: Path | None = None, play: bool = False):
    roster = os.environ.get("JUMPER_SUMO_LEAGUE", "")
    if not roster:
        raise ValueError("jumper.sumo.gyoji_policy: JUMPER_SUMO_LEAGUE (a roster) is required")
    cfg = sumo_env_cfg(asset=asset, play=play, gyoji=True)     # the west side's league too

    # ── The east side: a league, no learner ─────────────────────────────────
    from ..tactics.league import use_league

    cfg.actions.pop("joint_pos")
    cfg.commands["twist"] = HeldCommandCfg(initial=(0.0, 0.0, 0.0))
    cfg.commands["posture"] = HeldCommandCfg(initial=(0.0, 0.0, 0.0, _NEUTRAL_HEIGHT))
    speeds = os.environ.get("JUMPER_SUMO_OPP_SPEED")
    use_league(cfg, RING_RADIUS, roster, "robot",
               speed_range=tuple(float(v) for v in speeds.split(",")) if speeds else None)

    # ── The gyoji learns ─────────────────────────────────────────────────────
    base = cfg.actions[GYOJI]
    kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    kwargs.pop("policy", None)
    kwargs.pop("policy_group", None)
    cfg.actions[GYOJI] = LearnedGyojiActionCfg(**kwargs)
    # Put down about its shikiri spot, not exactly on it, so that it learns to
    # find its place from wherever it is at the tachiai.
    cfg.events["reset_gyoji_base"].params["pose_range"] = {
        "x": (-0.10, 0.10), "y": (-0.08, 0.08), "z": (0.01, 0.03), "yaw": (-0.4, 0.4)}
    gyoji_body = ContactMatch(mode="subtree", pattern="base_link", entity=GYOJI)
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + tuple(
        ContactSensorCfg(name=f"gyoji_touch_{side}", primary=gyoji_body,
                         secondary=ContactMatch(mode="subtree", pattern="base_link", entity=entity),
                         fields=("found",), reduce="none", num_slots=1)
        for side, entity in (("east", "robot"), ("west", OPPONENT)))

    # ── What it sees ─────────────────────────────────────────────────────────
    actor = gyoji_policy_terms(RING_RADIUS)
    critic = dict(actor, touched=ObservationTermCfg(func=gyoji_touched))
    cfg.observations = {
        "actor": ObservationGroupCfg(terms=actor),
        "critic": ObservationGroupCfg(terms=critic),
    }

    # ── What it is paid for ─────────────────────────────────────────────────
    r = {"radius": RING_RADIUS}
    funcs = {
        "watch_side": (watch_side, r), "watch_distance": (watch_distance, r), "facing": (facing, r),
        "back_to_front": (back_to_front, r), "too_close": (too_close, {}), "touching": (touching, {}),
        "beyond_straw": (beyond_straw, {}), "moving": (moving, {}), "turning": (turning, {}),
        "bobbing": (bobbing, {}), "action_rate": (action_rate_l2, {}),
        "crouch_at_the_straw": (crouch_at_the_straw, r), "standing_after": (standing_after, r),
        "hidden": (hidden_from_front, r), "saw_decision": (saw_decision, r),
    }
    # JUMPER_GYOJI_WEIGHTS="too_close=-4,touching=-8": weights for a variant run.
    weights = dict(WEIGHTS)
    for item in filter(None, os.environ.get("JUMPER_GYOJI_WEIGHTS", "").split(",")):
        name, value = item.split("=")
        if name not in weights:
            raise ValueError(f"JUMPER_GYOJI_WEIGHTS: no reward term {name!r}")
        weights[name] = float(value)
    cfg.rewards = {name: RewardTermCfg(func=f, weight=weights[name], params=dict(p))
                   for name, (f, p) in funcs.items()}
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.rewards["fell"] = RewardTermCfg(func=gyoji_ended, weight=-FALL_PENALTY / step_dt)

    # ── When an episode ends ────────────────────────────────────────────────
    cfg.terminations["gyoji_fell"] = TerminationTermCfg(
        func=gyoji_fell, params={"limit_angle": math.radians(50.0)})
    cfg.terminations["gyoji_off"] = TerminationTermCfg(func=gyoji_off)
    if not play:
        cfg.episode_length_s = 20.0
    return cfg
