"""Hyper-parameters for jumper.sumo.tactics.

A small network on a 28-wide observation with 6 outputs. Two departures from the
posture baseline, both because the decision is a bout and not a gait:

- **gamma 0.999** (a 5 s horizon at 200 Hz, against posture's 2 s): the prize is
  paid once, when a bout ends, and a bout lasts seconds.
- **No symmetry.** The baseline's mirror transform is written for the posture
  observation layout; this one is different.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ...common.ppo import jumper_ppo_baseline


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = _agent_cfg()
    import os

    if os.environ.get("JUMPER_TACTICS_WAZA", "") == "1":
        # Claws (waza.py): exploration noise capped. Unused outputs only feel the
        # entropy bonus, and in the first claw runs their noise grew to 3.3 (the
        # walking commands' to 4.4) -- noise, not exploration, on outputs that
        # are clipped to [-1, 1] anyway.
        cfg.actor.distribution_cfg["std_range"] = (0.05, 1.5)
    return cfg


def _agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.sumo.tactics",
        actor_hidden_dims=(256, 128, 64),
        critic_hidden_dims=(256, 128, 64),
        init_std=0.5,
        entropy_coef=0.005,
        learning_rate=5.0e-4,
        desired_kl=0.01,
        gamma=0.999,
        lam=0.95 ** 0.25,
        num_learning_epochs=5,
        num_mini_batches=4,
        num_steps_per_env=48,
        max_iterations=1_000,
        symmetry=False,
        use_data_augmentation=False,
        use_mirror_loss=False,
    )
