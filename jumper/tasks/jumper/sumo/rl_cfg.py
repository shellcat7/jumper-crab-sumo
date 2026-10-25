"""Hyper-parameters for jumper.sumo: posture's, under this task's name."""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..posture.rl_cfg import agent_cfg as posture_agent_cfg


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    cfg = posture_agent_cfg()
    cfg.experiment_name = "jumper.sumo"
    return cfg
