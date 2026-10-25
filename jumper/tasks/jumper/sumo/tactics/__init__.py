"""jumper.sumo.tactics -- a tactician that steers a frozen walking policy in the ring
(jumper-challenge-2026).

The learner here does not move joints: it chooses a velocity and a posture
command every step from what it sees of the opponent and the ring, and a frozen
posture policy walks it. See `env_cfg.py` and `mdp.py`.
"""

from __future__ import annotations

from ....registry import register
from ...common.assets import JUMPER_ASSETS

register(
    id="jumper.sumo.tactics",
    assets=JUMPER_ASSETS,
    description="sumo tactician: velocity and posture commands to a frozen walking policy, against a walking opponent",
    tags=("jumper", "sumo", "multi-robot", "hierarchical"),
)
