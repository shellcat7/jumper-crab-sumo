"""jumper.gesture_claw_twirl: an original arm choreography, written by
jumper-challenge-2026/tools/choreo/claw_twirl.py. Copied from gesture_hello
with the id, the clip and the tracking stds changed."""
from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.gesture_claw_twirl",
    assets=JUMPER_ASSETS,
    description="jumper hexapod imitating an original claw-twirl arm choreography; the clip is in tasks/jumper/gesture_claw_twirl/media/",
    tags=("imitation", "jumper", "gesture", "flat"),
)
