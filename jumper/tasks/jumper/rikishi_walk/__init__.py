"""jumper.rikishi_walk -- the wrestlers' legs: posture walking while one front claw at
a time is taken to a claw pose, as a tactician's claws take it (jumper-challenge-2026).
See `env_cfg.py`. A trained checkpoint is a tactician's walking policy
(JUMPER_TACTICS_LOW, JUMPER_SUMO_OPP_TACTICIAN_LOW).
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.rikishi_walk",
    assets=JUMPER_ASSETS,
    description="jumper posture walking with one front claw at a time held at a claw pose (reach, strike, raised)",
    tags=("jumper", "posture", "five_foot"),
)
