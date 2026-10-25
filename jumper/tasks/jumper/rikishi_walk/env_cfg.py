"""Environment config for jumper.rikishi_walk (jumper-challenge-2026).

The wrestlers' legs. A tactician's claws (`sumo/tactics/waza.py`) are its front
feet: to reach, grip or strike, one front arm leaves the clay and is laid over the
walking policy's targets -- and the walking policy, the official posture example,
was trained on six feet. The gyoji showed what that costs: on the official legs
with one arm held up it bobbed 5.5 mm, on legs trained for it (`jumper.gyoji_walk`)
2.4 mm, as much as a wrestler walking on six. The claw wrestlers stagger the same
way, and the claw-lean runs of 10/13-14 say the claws cost them bouts.

This is `jumper.posture` where, every few seconds, one front arm (never both: the
claw rule, `waza.py`) is taken from the policy and swung to one of the claw's poses
-- reach for the shell, reach low for the legs, the finish of a thrust, of a slap,
or raised as in the shikiri -- or both arms are given back. The policy still sends
20 targets, so a checkpoint walks a tactician unchanged (`JUMPER_TACTICS_LOW`,
`JUMPER_SUMO_OPP_TACTICIAN_LOW`, a league's walking policy).

    1. the claws      `ClawJointPositionAction`: per environment, an arm and a pose
                      drawn every `HOLD_S` seconds (none in `FREE_SHARE` of draws),
                      eased in and out over about 0.1 s as the tactician's are
    2. the gait       the tripod while both claws are down; with one up, the five-
                      legged gait of `jumper.gyoji_walk` (RM+LR / LM+RR, the other
                      front claw planted)
    3. stance terms   the tripod's `stance_load` and `stance_ground_gap`: off
    4. the bob        the base's vertical speed and roll / pitch rates, a cost
    5. disturbances   meant to be trained with wind (JUMPER_WIND_N, _CHANGE_S,
                      `common/experiment.py`), from the wind-trained walker
                      (posture_wind12_1000, the steadiest under wind and pushes)
    6. the oshi       (JUMPER_OSHI_N, 10/15) the push of an opponent: a horizontal
                      force on the base that turns with the robot, from the front
                      within `OSHI_CONE` of straight on (from any side in `OSHI_ANY`
                      of draws), 0..N newtons, re-drawn every JUMPER_OSHI_CHANGE_S
                      seconds (default 1,4). The wind blows from a fixed direction
                      however the robot turns; a wrestler is pushed from where it
                      faces, and it has to drive into the push rather than lean
                      against a steady breeze. Written every step to the base's
                      `xfrc_applied`, so it replaces the wind (do not set both).
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..common.mdp.rewards import contact_state, cycling_gate, moving_gate, phase_match
from ..gyoji_walk.env_cfg import BOB_WEIGHT, FIVE, SWING_A, SWING_B, body_bob
from ..posture.env_cfg import env_cfg as posture_env_cfg
from ..tripod.mdp.phase import GAIT_FREQ_HZ, gait_phase
from ..tripod.mdp.rewards import tripod_phase_score

#: Left-arm poses (J0..J3) from `waza.py`; the right arm mirrors J0, J2, J3.
CLAW_POSES = (
    (-0.6, -2.0, 0.0, -1.3),      # REACH: the claw tip at the shell
    (-0.8, -1.2, 0.0, -0.95),     # REACH_LOW: at the legs
    (-1.6, -3.4, 0.0, -0.6),      # tsuki, the finish
    (-1.6, -2.6, 0.0, -0.95),     # harite, the finish
    (-0.35, -2.0, -1.0, -1.6),    # arms.RAISED, the shikiri's
)
FREE_SHARE = 0.4                  # draws with both claws down
HOLD_S = (0.8, 3.0)               # seconds an arm and pose are held
EASE_S = 0.1
#: The five-legged contact columns with the left claw up: RF, LM, RM, LR, RR.
FIVE_LEFT_UP = (1, 2, 3, 4, 5)
#: The oshi (see the module docstring): newtons (0: off), seconds between draws,
#: the cone around straight on, and the share of draws from any side.
OSHI_N = float(os.environ.get("JUMPER_OSHI_N", "") or 0.0)
OSHI_CHANGE_S = tuple(float(v) for v in (os.environ.get("JUMPER_OSHI_CHANGE_S", "") or "1,4").split(","))
OSHI_CONE = math.radians(45.0)
OSHI_ANY = 0.25


@dataclass(kw_only=True)
class ClawJointPositionActionCfg(JointPositionActionCfg):
    def build(self, env) -> "ClawJointPositionAction":
        return ClawJointPositionAction(self, env)


class ClawJointPositionAction(JointPositionAction):
    """Posture's joint action with one front arm at a time taken to a claw pose."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        names = list(self.target_names)
        self._cols = torch.tensor(
            [[next(i for i, nm in enumerate(names) if nm.endswith(f"{s}F_J{k}_joint")) for k in range(4)]
             for s in "LR"], device=dev)                                   # [2, 4]
        left = torch.tensor(CLAW_POSES, device=dev)                        # [P, 4]
        right = left * torch.tensor([-1.0, 1.0, -1.0, -1.0], device=dev)
        self._poses = torch.stack((left, right))                            # [2, P, 4]
        self._side = torch.full((n,), -1, dtype=torch.long, device=dev)     # -1 none, 0 L, 1 R
        self._pose = torch.zeros(n, dtype=torch.long, device=dev)
        self._timer = torch.zeros(n, device=dev)
        self._w = torch.zeros(n, 2, device=dev)                             # per arm, eased
        self._alpha = min(1.0, env.step_dt / EASE_S)
        # Training (where the pushes are), or a replay that keeps its disturbances.
        self._oshi = OSHI_N > 0.0 and ("push_robot" in env.cfg.events
                                       or os.environ.get("JUMPER_PLAY_DISTURB", "") == "1")
        if self._oshi:
            self._base_ids = self._entity.find_bodies("base_link")[0]
            self._oshi_mag = torch.zeros(n, device=dev)
            self._oshi_dir = torch.zeros(n, device=dev)                       # from the heading
            self._oshi_timer = torch.zeros(n, device=dev)

    def _draw(self, ids: torch.Tensor) -> None:
        k = len(ids)
        free = torch.rand(k, device=self.device) < FREE_SHARE
        side = torch.randint(0, 2, (k,), device=self.device)
        self._side[ids] = torch.where(free, torch.full_like(side, -1), side)
        self._pose[ids] = torch.randint(0, self._poses.shape[1], (k,), device=self.device)
        self._timer[ids] = torch.empty(k, device=self.device).uniform_(*HOLD_S)

    def _draw_oshi(self, ids: torch.Tensor) -> None:
        k = len(ids)
        any_side = torch.rand(k, device=self.device) < OSHI_ANY
        cone = (torch.rand(k, device=self.device) * 2.0 - 1.0) * OSHI_CONE
        # The force points backwards (heading + pi) for a push from the front.
        self._oshi_dir[ids] = torch.where(any_side, torch.rand(k, device=self.device) * 2.0 * math.pi,
                                          math.pi + cone)
        self._oshi_mag[ids] = torch.rand(k, device=self.device) * OSHI_N
        self._oshi_timer[ids] = torch.empty(k, device=self.device).uniform_(*OSHI_CHANGE_S)

    def _apply_oshi(self) -> None:
        self._oshi_timer -= self._env.step_dt
        due = (self._oshi_timer <= 0).nonzero().flatten()
        if len(due):
            self._draw_oshi(due)
        angle = self._entity.data.heading_w + self._oshi_dir
        forces = torch.zeros((self.num_envs, len(self._base_ids), 3), device=self.device)
        forces[:, :, 0] = (self._oshi_mag * torch.cos(angle))[:, None]
        forces[:, :, 1] = (self._oshi_mag * torch.sin(angle))[:, None]
        self._entity.write_external_wrench_to_sim(forces, torch.zeros_like(forces),
                                                  body_ids=self._base_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        if self._oshi:
            self._apply_oshi()
        self._timer -= self._env.step_dt
        due = (self._timer <= 0).nonzero().flatten()
        if len(due):
            self._draw(due)
        target = torch.stack(((self._side == 0).float(), (self._side == 1).float()), -1)
        self._w = self._w + self._alpha * (target - self._w)
        p = self._processed_actions
        for c in (0, 1):
            cols = self._cols[c]
            pose = self._poses[c][self._pose]                               # [N, 4]
            w = self._w[:, c:c + 1]
            p[:, cols] = p[:, cols] + w * (pose - p[:, cols])

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, device=self.device))
        self._w[ids] = 0.0
        self._draw(ids)
        if self._oshi:
            self._draw_oshi(ids)

    def claw_up(self) -> torch.Tensor:
        """[N] -1 both claws down, 0 the left up, 1 the right (eased past half)."""
        up = self._w > 0.5
        return torch.where(up[:, 0], 0, torch.where(up[:, 1], 1, -1))


def claw_gait(env, sensor_name: str, command_name: str, command_threshold: float = 0.05,
              sigma: float = 0.7, max_air_time: float = 0.5, max_contact_time: float = 1.0,
              freq_hz: float = GAIT_FREQ_HZ) -> torch.Tensor:
    """The tripod with both claws down; with one up, the five-legged gait, the
    other front claw planted (`gyoji_walk.gunbai_phase_score`, mirrored for the left)."""
    contact = contact_state(env, sensor_name)
    phase = gait_phase(env, freq_hz)
    up = env.action_manager.get_term("joint_pos").claw_up()
    tripod = tripod_phase_score(contact, phase, sigma)

    def five(cols):
        c = contact[:, list(cols)]
        return torch.where(phase < 0.5, phase_match(c, SWING_A, sigma), phase_match(c, SWING_B, sigma))

    score = torch.where(up == 1, five(FIVE), torch.where(up == 0, five(FIVE_LEFT_UP), tripod))
    return (score * moving_gate(env, command_name, command_threshold)
            * cycling_gate(env, sensor_name, max_air_time, max_contact_time))


def env_cfg(asset: Path | None = None, play: bool = False):
    cfg = posture_env_cfg(asset=asset, play=play)
    base = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = ClawJointPositionActionCfg(
        **{f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init})
    gait = cfg.rewards.pop("tripod_gait")
    cfg.rewards["claw_gait"] = RewardTermCfg(func=claw_gait, weight=gait.weight, params=dict(gait.params))
    for name in ("stance_load", "stance_ground_gap"):
        cfg.rewards.pop(name, None)
    cfg.rewards["body_bob"] = RewardTermCfg(
        func=body_bob, weight=float(os.environ.get("JUMPER_RIKISHI_BOB", BOB_WEIGHT)))
    return cfg
