"""Hyper-parameters for jumper.rikishi_walk: jumper.posture's, at half the learning
rate (fine-tuned from the wind-trained walker), without the mirror: the claw drawn
for a step is the left or the right, and a mirrored sample would pair the left
arm's pose with the right arm's columns.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline
from ..posture.rl_cfg import NUM_STEPS_PER_ENV
from ..posture.rl_cfg import runner_cls as runner_cls  # noqa: F401  (the velocity runner)


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.rikishi_walk",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=1.0,
        entropy_coef=0.0015,
        learning_rate=5.0e-4,
        desired_kl=0.01,
        gamma=0.99 ** 0.25,
        lam=0.95 ** 0.25,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        max_iterations=1_500,
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
    )
