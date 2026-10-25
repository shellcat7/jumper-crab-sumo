"""Hyper-parameters for jumper.gyoji_walk: jumper.posture's, without the mirror.

The posture baseline augments with the left-right mirror; the gyoji is not
symmetric any more (its right-front arm is held up with the gunbai), so a mirrored
sample would teach the legs a robot that does not exist. Fine-tuned from the
official posture example (model_74800), at half posture's learning rate.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline
from ..posture.rl_cfg import NUM_STEPS_PER_ENV
from ..posture.rl_cfg import runner_cls as runner_cls  # noqa: F401  (the velocity runner)


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.gyoji_walk",
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
