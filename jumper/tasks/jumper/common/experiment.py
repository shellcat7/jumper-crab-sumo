"""Experiment knobs for jumper-challenge-2026, read from environment variables.

Unset, every knob leaves the task exactly as the official configuration has it, so
an official checkpoint trains and replays unchanged. Nothing here changes an
observation or an action, so `train.py --checkpoint` can continue an official run
with any of these set.

    JUMPER_PUSH_SCALE     multiply `push_robot`'s velocity range (x, y, z and the
                          rotations) by this factor: stronger kicks to stand against
    JUMPER_WIND_N         a steady horizontal force on the base, in newtons, as a
                          stand-in for wind: on every reset, and every
                          JUMPER_WIND_CHANGE_S seconds if set, each environment draws
                          a direction and a magnitude up to this. Wind on a real body
                          scales with its exposed area; this is a force at the base.
    JUMPER_WIND_CHANGE_S  "a,b": re-draw the wind every a..b seconds (gusts that turn)
    JUMPER_PLAY_DISTURB   "1": keep the pushes and the wind in replay (play.py), which
                          otherwise turns disturbances off, for demonstrations
    JUMPER_TWIST_RESAMPLE_S  "a,b": training only -- re-draw the velocity command every
                          a..b seconds instead of the official 3..8, for a walker
                          that answers a tactician's or a gyoji's quickly changing
                          commands (10/15: they change them every step, smoothed)

The speed knob is `JUMPER_LIN_CEILING` in `tasks/jumper/posture/env_cfg.py`, because
the posture task derives its gait frequency from the ceiling at import time.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp
from mjlab.envs.mdp.events import resolve_env_ids
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg



def _base() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=("base_link",))


def _float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def _pair(name: str) -> tuple[float, float] | None:
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    a, b = (float(v) for v in value.split(","))
    return (a, b)


def horizontal_wind(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    max_force: float,
    asset_cfg: SceneEntityCfg | None = None,
) -> None:
    """A steady horizontal force on the base: random direction, magnitude 0..max.

    Written to `xfrc_applied`, which MuJoCo holds until it is overwritten, so it
    stays on until the next draw (the next reset or the next interval).
    """
    env_ids = resolve_env_ids(env, env_ids)
    asset = env.scene[asset_cfg.name]
    n = len(env_ids)
    heading = torch.rand(n, device=env.device) * 2.0 * math.pi
    magnitude = torch.rand(n, device=env.device) * max_force
    forces = torch.zeros((n, len(asset_cfg.body_ids), 3), device=env.device)
    forces[:, :, 0] = (magnitude * torch.cos(heading))[:, None]
    forces[:, :, 1] = (magnitude * torch.sin(heading))[:, None]
    torques = torch.zeros_like(forces)
    asset.write_external_wrench_to_sim(forces, torques, env_ids=env_ids,
                                       body_ids=asset_cfg.body_ids)


def apply_experiment_knobs(cfg: ManagerBasedRlEnvCfg, play: bool) -> list[str]:
    """Apply whichever knobs are set; return a description of each one applied."""
    from .constants import PUSH_VELOCITY_RANGE
    applied: list[str] = []
    resample = _pair("JUMPER_TWIST_RESAMPLE_S")
    if resample is not None and not play and "twist" in cfg.commands:
        cfg.commands["twist"].resampling_time_range = resample
        applied.append(f"velocity command re-drawn every {resample} s")
    disturb = (not play) or os.environ.get("JUMPER_PLAY_DISTURB", "") == "1"
    if not disturb:
        if applied:
            print("[experiment] " + "; ".join(applied))
        return applied

    scale = _float("JUMPER_PUSH_SCALE")
    if scale is not None:
        push = cfg.events.get("push_robot")
        if push is None:  # replay removes it; put the training one back for the demo
            # Training's push is mjlab's velocity default (velocity_env_cfg.py: 1-3 s,
            # rotations included) with x/y/z replaced by PUSH_VELOCITY_RANGE, as
            # velocity_env.py does. Leaving the rotations out made replay pushes
            # weaker than training's (10/9: 1.7 deg of tilt under x10).
            training_range = {"roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
                              **PUSH_VELOCITY_RANGE}
            push = EventTermCfg(func=mdp.push_by_setting_velocity, mode="interval",
                                interval_range_s=(1.0, 3.0),
                                params={"velocity_range": training_range})
        if push is not None:
            push.params["velocity_range"] = {
                k: (lo * scale, hi * scale) for k, (lo, hi) in push.params["velocity_range"].items()
            }
            cfg.events["push_robot"] = push
            applied.append(f"push x{scale:g}: {push.params['velocity_range']}")

    wind = _float("JUMPER_WIND_N")
    if wind is not None and wind > 0:
        cfg.events["wind"] = EventTermCfg(
            func=horizontal_wind, mode="reset", params={"max_force": wind, "asset_cfg": _base()})
        change = _pair("JUMPER_WIND_CHANGE_S")
        if change is not None:
            cfg.events["wind_change"] = EventTermCfg(
                func=horizontal_wind, mode="interval", interval_range_s=change,
                params={"max_force": wind, "asset_cfg": _base()})
        applied.append(f"wind up to {wind:g} N" + (f", re-drawn every {change} s" if change else ""))

    if applied:
        print("[experiment] " + "; ".join(applied))
    return applied
