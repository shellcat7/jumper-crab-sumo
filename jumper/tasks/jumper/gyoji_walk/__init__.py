"""jumper.gyoji_walk -- the gyoji's legs: posture walking on five legs, the
right-front arm holding the gunbai (jumper-challenge-2026). See `env_cfg.py`.
A trained checkpoint walks the gyoji in any sumo replay with
JUMPER_SUMO_GYOJI_WALKER=<model.pt>.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gyoji_walk",
    assets=JUMPER_ASSETS,
    description="jumper posture walking on five legs, the right-front arm holding the gyoji's gunbai",
    tags=("jumper", "posture", "five_foot"),
)
