"""jumper.sumo.tactics: the sumo ring with a tactician in charge of the robot.

`jumper.sumo` trains the walking policy itself to push, and the only thing it is
told is "walk at the opponent": it cannot choose to side-step, circle or give
way, because following the command is what it is paid for. Here the walking is
frozen and the choice is what is learned:

- **The robot's action** is `mdp.TacticsAction`: six numbers mapped onto a
  velocity and a posture command, smoothed, and handed to a frozen walking
  policy (`LOW_LEVEL_CHECKPOINT`, or `JUMPER_TACTICS_LOW`) that runs inside the
  environment exactly as the opponent's does. The robot's original 415-wide
  observation becomes that policy's private observation.
- **The tactician's observation** (`mdp.py`): its own motion, the opponent's
  position, heading, velocity and tilt in its frame, the ring, and the bout's
  phase. The critic also sees whether the two are touching.
- **Rewards**: the bout (win +30, loss -30, `jumper.sumo`'s), pushing the
  opponent outwards, a penalty for standing near the straw, a small pull to stay
  engaged after the tachiai, and a penalty on jerky choices. The posture task's
  walking rewards are gone: walking is not this learner's job.
- **Reading the opponent** (`JUMPER_TACTICS_TREND=1`): a moving average of how
  the opponent has been moving (`mdp.opponent_trend`), for runs against a pool
  of opponents. It changes the observation width, so a checkpoint trained with
  it replays only with it.
- **Styles** (`JUMPER_TACTICS_STYLE=mawari`): a dense term for how a bout is
  worked, not only how it ends. `mawari` pays for being to the opponent's side
  or back while touching it (`mdp.behind`), the way to okuridashi.
- **Technique coaches** (`JUMPER_TACTICS_COACH=oshidashi,okuridashi,...`): a
  bonus on top of the win when the referee names one of those kimarite, so a
  run can be steered towards a technique and its share measured.
- **Claws** (`JUMPER_TACTICS_WAZA=1`, `waza.py`): seven more outputs -- each
  claw's opening (an open claw reaches for the mawashi; closed there, it holds
  it), strikes with either arm (tsuki, harite, hataki, kachiage) and their speed,
  and how low to crouch in the shikiri -- and what it knows of them (16 inputs:
  claw tips at the opponent's mawashi, grips both ways, strikes, and the kind of
  wrestler it is being trained as, `waza.STYLES`, drawn per bout from
  `JUMPER_TACTICS_RIKISHI`). Paid by style: grips held, strikes landed, an
  opponent turned while held, and a bonus for winning by the style's kimarite.
- Everything else -- the opponent, the ring, the shikiri and tachiai, the gyoji
  and referee when asked for -- is `jumper.sumo`'s.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from mjlab.envs.mdp.observations import last_action
from mjlab.envs.mdp.rewards import action_rate_l2
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ..env_cfg import (
    _NEUTRAL_HEIGHT, OPPONENT_CHECKPOINT, RING_RADIUS, _add_referee_terms,
)
from ..env_cfg import env_cfg as sumo_env_cfg
from ..opponent import HeldCommandCfg, opponent_observation_group
from ..referee import KIMARITE, referee_watch
from .mdp import (
    TacticsActionCfg, behind, bout_phase, closing, in_contact, kimarite_won, near_straw, opponent_relative,
    opponent_trend,
    own_motion, ring_state,
)

TACTICS = "tactics"
#: A technique coach's bonus, on top of the win's 30 (one step, so / step_dt).
COACH_BONUS = 15.0
#: A style term's weight, per second of the style (the bout's +30 is once).
STYLE_WEIGHT = 2.0
#: Claws (`waza.py`): a strike that lands, once; a strike begun, once (a cost);
#: a win by the style's own kimarite, once (the win's 30 on top).
STRIKE_HIT_PAY = 0.5
STRIKE_COST = 0.05
WAZA_KIMARITE_BONUS = 15.0
#: The walking policy the tactician steers. The official posture example by
#: default (it tracks the whole command range); `JUMPER_TACTICS_LOW` swaps in
#: another, e.g. a `jumper.sumo`-trained one that already knows how to push.
LOW_LEVEL_CHECKPOINT = OPPONENT_CHECKPOINT


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = sumo_env_cfg(asset=asset, play=play)

    # ── The walking policy keeps the robot's own observation, privately ────
    low_level = opponent_observation_group(
        cfg.observations["actor"], entity="robot", twist="twist", posture="posture",
        action=TACTICS)
    cfg.commands["twist"] = HeldCommandCfg(initial=(0.0, 0.0, 0.0))
    cfg.commands["posture"] = HeldCommandCfg(initial=(0.0, 0.0, 0.0, _NEUTRAL_HEIGHT))
    cfg.curriculum = {}
    base = cfg.actions.pop("joint_pos")
    kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    checkpoint = os.environ.get("JUMPER_TACTICS_LOW") or str(LOW_LEVEL_CHECKPOINT)
    cfg.actions = {
        TACTICS: TacticsActionCfg(
            **kwargs, checkpoint=checkpoint, observation_group=low_level,
            twist_name="twist", posture_name="posture", neutral_height=_NEUTRAL_HEIGHT),
        **cfg.actions,
    }

    # ── What the tactician sees ─────────────────────────────────────────────
    if not any(s.name == "bout_contact" for s in (cfg.scene.sensors or ())):
        _add_referee_terms(cfg)
    actor = {
        "own_motion": ObservationTermCfg(func=own_motion),
        "opponent": ObservationTermCfg(func=opponent_relative),
        "ring": ObservationTermCfg(func=ring_state, params={"radius": RING_RADIUS}),
        "bout": ObservationTermCfg(func=bout_phase),
        "actions": ObservationTermCfg(func=last_action),
    }
    if os.environ.get("JUMPER_TACTICS_TREND", "") == "1":
        actor["trend"] = ObservationTermCfg(func=opponent_trend)
    waza_on = os.environ.get("JUMPER_TACTICS_WAZA", "") == "1"
    if waza_on:
        from .waza import claws_state

        actor["claws"] = ObservationTermCfg(func=claws_state)
    if os.environ.get("JUMPER_TACTICS_MEMORY", "") == "1":
        # About a second of the bout as it was (mdp.bout_memory): last, so a
        # checkpoint widened for it (tools/train/expand_memory.py) keeps its columns.
        from .mdp import bout_memory

        actor["memory"] = ObservationTermCfg(func=bout_memory)
    critic = dict(actor, contact=ObservationTermCfg(func=in_contact))
    cfg.observations = {
        "actor": ObservationGroupCfg(terms=actor),
        "critic": ObservationGroupCfg(terms=critic),
    }

    # ── What it is paid for ─────────────────────────────────────────────────
    bout_terms = {k: cfg.rewards[k] for k in ("sumo_win", "sumo_loss", "push_out") if k in cfg.rewards}
    cfg.rewards = {
        **bout_terms,
        "near_straw": RewardTermCfg(func=near_straw, weight=-2.0, params={"radius": RING_RADIUS}),
        "closing": RewardTermCfg(func=closing, weight=0.5),
        "action_rate": RewardTermCfg(func=action_rate_l2, weight=-0.02),
    }
    style = os.environ.get("JUMPER_TACTICS_STYLE", "")
    if style == "mawari":
        cfg.rewards["behind"] = RewardTermCfg(func=behind, weight=STYLE_WEIGHT)
    elif style:
        raise ValueError(f"JUMPER_TACTICS_STYLE: unknown style {style!r}")
    coach = tuple(n for n in os.environ.get("JUMPER_TACTICS_COACH", "").split(",") if n)
    if coach:
        unknown = set(coach) - {n for n, _ in KIMARITE}
        assert not unknown, f"JUMPER_TACTICS_COACH: no such kimarite {sorted(unknown)}"
        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        if "referee" not in cfg.terminations and "bout_over" not in cfg.terminations:
            cfg.terminations["referee"] = TerminationTermCfg(
                func=referee_watch, params={"radius": RING_RADIUS})
        cfg.rewards["technique"] = RewardTermCfg(
            func=kimarite_won, weight=COACH_BONUS / step_dt,
            params={"names": coach, "radius": RING_RADIUS})
    variety = float(os.environ.get("JUMPER_TACTICS_VARIETY", "0"))
    if variety > 0.0:
        # A win pays more the rarer its kimarite has been (mdp.rare_kimarite_won, 10/20).
        from .mdp import rare_kimarite_won

        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        if "referee" not in cfg.terminations and "bout_over" not in cfg.terminations:
            cfg.terminations["referee"] = TerminationTermCfg(
                func=referee_watch, params={"radius": RING_RADIUS})
        cfg.rewards["variety"] = RewardTermCfg(
            func=rare_kimarite_won, weight=variety / step_dt, params={"radius": RING_RADIUS})
    comeback = float(os.environ.get("JUMPER_TACTICS_COMEBACK", "0"))
    if comeback > 0.0:
        # A win pays more the nearer the straw the winner had been driven (mdp.comeback_won, 10/21).
        from .mdp import comeback_won

        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        if "referee" not in cfg.terminations and "bout_over" not in cfg.terminations:
            cfg.terminations["referee"] = TerminationTermCfg(
                func=referee_watch, params={"radius": RING_RADIUS})
        cfg.rewards["comeback"] = RewardTermCfg(
            func=comeback_won, weight=comeback / step_dt, params={"radius": RING_RADIUS})
    if waza_on:
        from .waza import (
            grip_held, strike_begun, strike_landed, style_kimarite_won, turned_while_held,
        )

        step_dt = cfg.sim.mujoco.timestep * cfg.decimation
        if "referee" not in cfg.terminations and "bout_over" not in cfg.terminations:
            cfg.terminations["referee"] = TerminationTermCfg(
                func=referee_watch, params={"radius": RING_RADIUS})
        cfg.rewards["grip"] = RewardTermCfg(func=grip_held, weight=1.0)
        cfg.rewards["strike_hit"] = RewardTermCfg(func=strike_landed, weight=STRIKE_HIT_PAY / step_dt)
        cfg.rewards["strike_cost"] = RewardTermCfg(func=strike_begun, weight=-STRIKE_COST / step_dt)
        cfg.rewards["throw_turn"] = RewardTermCfg(func=turned_while_held, weight=1.0)
        cfg.rewards["style_kimarite"] = RewardTermCfg(
            func=style_kimarite_won, weight=WAZA_KIMARITE_BONUS / step_dt,
            params={"radius": RING_RADIUS})
        claw_bonus = float(os.environ.get("JUMPER_WAZA_CLAW_BONUS", "0"))
        if claw_bonus > 0.0:
            # A win by the claws (a leg or arm held, a strike, a throw) paid on top.
            from .waza import claw_kimarite_won

            cfg.rewards["claw_kimarite"] = RewardTermCfg(
                func=claw_kimarite_won, weight=claw_bonus / step_dt, params={"radius": RING_RADIUS})
    if not play:
        cfg.episode_length_s = 20.0
    return cfg
