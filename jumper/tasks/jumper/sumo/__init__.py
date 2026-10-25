"""jumper.sumo -- two Jumpers in a ring (jumper-challenge-2026).

The learner is the posture robot; the opponent is a second Jumper driven by a frozen
posture policy that walks at it. See `env_cfg.py` and `opponent.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.sumo",
    assets=JUMPER_ASSETS,
    description="two jumper hexapods in a ring; the opponent walks at the learner on a frozen posture policy",
    tags=("jumper", "sumo", "multi-robot", "posture"),
)
