"""jumper.gesture_claw_twirl: an original arm choreography, written by
jumper-challenge-2026/tools/choreo/claw_twirl.py. Copied from gesture_hello
with the id, the clip and the tracking stds changed."""
from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    """This task's rsl_rl config: `logs/<model>/jumper.gesture_claw_twirl/<date-time>`."""
    return jumper_ppo_baseline(
        experiment_name="jumper.gesture_claw_twirl",
        # Off, and not a tuning choice: the gesture is one-handed, and mirroring its
        # samples would train the average of a left-handed and a right-handed wave.
        # `symmetry.py` would also refuse the reference terms.
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        # Half the baseline's: imitation needs less exploration than locomotion,
        # and a rising `Policy/mean_std` costs precision terms first.
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=24,
        max_iterations=10_000,
    )


def runner_cls() -> type:
    """mjlab's motion-tracking runner, as jumper.dance: it logs the tracking metrics
    and the adaptive sampler's entropy, and bakes the clip into the exported ONNX."""
    from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

    return MotionTrackingOnPolicyRunner
