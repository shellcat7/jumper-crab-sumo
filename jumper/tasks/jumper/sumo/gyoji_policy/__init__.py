"""jumper.sumo.gyoji_policy -- the gyoji learns where to stand and how to move
(jumper-challenge-2026).

Two frozen wrestlers drawn from a roster fight; the gyoji, the third Jumper,
chooses its velocity and crouch for a frozen walking policy, paid for watching
the bout as a gyoji must. See `env_cfg.py` and `mdp.py`. A trained checkpoint
referees any replay with JUMPER_SUMO_GYOJI=1 JUMPER_SUMO_GYOJI_POLICY=<model.pt>.
"""

from __future__ import annotations

from ....registry import register
from ...common.assets import JUMPER_ASSETS

register(
    id="jumper.sumo.gyoji_policy",
    assets=JUMPER_ASSETS,
    description="sumo gyoji: velocity and crouch commands to a frozen walking policy, refereeing two frozen wrestlers",
    tags=("jumper", "sumo", "multi-robot", "hierarchical"),
)
