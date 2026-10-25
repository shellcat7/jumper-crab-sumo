"""Hyper-parameters for jumper.sumo.gyoji_policy.

The tactician's network and PPO settings (`tactics/rl_cfg.py`), with a shorter
horizon: the gyoji is paid every step for where it stands, not once at the end
of the bout, so gamma 0.995 (1 s at 200 Hz) -- long enough to see a wrestler
coming at it, or the bout turning, before it is there.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = jumper_ppo_baseline(
        experiment_name="jumper.sumo.gyoji_policy",
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        init_std=0.6,
        entropy_coef=0.005,
        learning_rate=5.0e-4,
        desired_kl=0.01,
        gamma=0.995,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=48,
        max_iterations=1_500,
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
    )
    cfg.actor.distribution_cfg["std_range"] = (0.05, 1.5)
    return cfg
