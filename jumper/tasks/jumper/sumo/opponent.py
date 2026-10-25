"""The second Jumper in the ring, driven by a frozen walking policy.

jumper-challenge-2026. The learning robot is the task's `robot` entity, exactly as
in `jumper.posture`: same observation, same 20 actions, so a posture checkpoint
loads into it unchanged. The opponent is a second copy of the robot, `opponent`,
whose joints are driven from inside the environment by a frozen copy of a posture
policy. Nothing about it reaches the learner's action space:

- `OpponentPolicyAction` is an action term with `action_dim == 0` as far as the
  action manager is concerned, so the policy still outputs 20 numbers. Once per
  control step it computes the opponent's observation with an observation manager
  of its own, runs the frozen actor and hands the 20 results to the ordinary
  joint-position path (scale, offset, encoder bias) for the opponent's joints.
- Its observation is the posture actor group with every reference to the robot
  swapped for the opponent (`opponent_observation_group`). It has to be a separate
  manager: the strided-history terms advance once per environment step, and the
  env's own manager computing a second copy of them would be a second caller.
- The posture task keeps one gait clock per environment (`cadence.ENV_ATTR`), so
  the opponent gets a private `CadenceClock` instead (`OpponentGaitClock`).
- Its velocity and posture commands are `HeldCommand` terms: tensors the action
  term writes every step (walk at the robot, stand neutral), registered with the
  command manager so every reader finds them by name as it would the robot's.
- **A pool of opponents** (`checkpoint` holding several paths, comma-separated):
  each environment draws one of them at every reset, and an approach speed from
  `approach_speed_range` if one is given, so a learner meets more than one
  style of wrestler. Every policy in the pool runs on every environment and
  each environment keeps its own's output -- a handful of small MLPs.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ..posture.mdp.cadence import CadenceClock, VariableGaitClock

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

OPPONENT = "opponent"
TWIST_OPP = "twist_opp"
POSTURE_OPP = "posture_opp"


# ── A command that is whatever was last written into it ──────────────────────


class HeldCommand(CommandTerm):
    """A command tensor that never resamples; its owner writes it."""

    cfg: "HeldCommandCfg"

    def __init__(self, cfg: "HeldCommandCfg", env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._value = torch.tensor(cfg.initial, device=self.device).repeat(self.num_envs, 1)

    @property
    def command(self) -> torch.Tensor:
        return self._value

    def set(self, value: torch.Tensor) -> None:
        self._value[:] = value

    def _update_metrics(self) -> None:
        pass

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        self._value[env_ids] = torch.tensor(self.cfg.initial, device=self.device)

    def _update_command(self, env_ids: torch.Tensor | None) -> None:
        pass


@dataclass(kw_only=True)
class HeldCommandCfg(CommandTermCfg):
    initial: tuple[float, ...]
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

    def build(self, env: "ManagerBasedRlEnv") -> HeldCommand:
        return HeldCommand(self, env)


# ── The opponent's gait clock ─────────────────────────────────────────────────


class OpponentGaitClock(VariableGaitClock):
    """`VariableGaitClock` on a private `CadenceClock` rather than the env's one."""

    def __init__(self, cfg, env: "ManagerBasedRlEnv") -> None:  # noqa: D107
        params = cfg.params
        self._command = params["command_name"]
        self._threshold = float(params.get("command_threshold", 0.05))
        self.clock = CadenceClock(
            env, params["command_name"], float(params["stride"]),
            float(params["freq_min"]), float(params["freq_max"]),
            float(params["turn_radius"]),
        )


# ── The frozen actor ──────────────────────────────────────────────────────────


class FrozenActor(nn.Module):
    """rsl_rl's MLP actor rebuilt from a checkpoint: normalise, then ELU MLP.

    `EmpiricalNormalization.forward` is `(x - mean) / (std + eps)` with eps 1e-2;
    the mean action is the MLP output (the distribution's std is not used).
    """

    EPS = 1.0e-2

    def __init__(self, checkpoint: Path, device: str) -> None:
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
        self.register_buffer("mean", state["obs_normalizer._mean"])
        self.register_buffer("std", state["obs_normalizer._std"])
        linear = sorted(
            {int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")}
        )
        layers: list[nn.Module] = []
        for n, i in enumerate(linear):
            w = state[f"mlp.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(state[f"mlp.{i}.bias"])
            layers.append(lin)
            if n < len(linear) - 1:
                layers.append(nn.ELU())
        self.mlp = nn.Sequential(*layers)
        self.to(device)
        self.eval()

    @property
    def obs_dim(self) -> int:
        return int(self.mean.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.mlp[-1].out_features)

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.mean) / (self.std + self.EPS))


# ── The opponent's observation: the actor group, pointed at the opponent ─────


def opponent_observation_group(actor_group, entity: str = OPPONENT, twist: str = TWIST_OPP,
                               posture: str = POSTURE_OPP, action: str = OPPONENT):
    """Deep-copy the posture actor group and point every term at `entity`."""
    group = copy.deepcopy(actor_group)
    for name, term in group.terms.items():
        p = term.params
        for key, value in list(p.items()):
            if isinstance(value, SceneEntityCfg) and value.name == "robot":
                p[key] = dataclasses.replace(value, name=entity)
        if p.get("sensor_name", "").startswith("robot/"):
            p["sensor_name"] = entity + "/" + p["sensor_name"][len("robot/"):]
        if p.get("command_name") == "twist":
            p["command_name"] = twist
        if p.get("command_name") == "posture":
            p["command_name"] = posture
        if name == "projected_gravity":
            p["asset_cfg"] = SceneEntityCfg(entity)
        if name == "actions":
            # `last_action` with no name is the env's whole action vector, i.e.
            # the learner's; the opponent's own is its term's raw action.
            p["action_name"] = action
        if name == "gait_phase":
            term.func = OpponentGaitClock
        # Anything still naming the robot would silently observe the wrong body.
        # (Pointed at the robot itself -- the tactician's walking policy -- the
        # names that stay are the right ones.)
        stale = {n for n, kept in (("twist", twist), ("posture", posture)) if kept != n}
        moved = entity != "robot"
        leftovers = [k for k, v in p.items()
                     if (moved and isinstance(v, SceneEntityCfg) and v.name == "robot")
                     or (isinstance(v, str) and (v in stale or (moved and v.startswith("robot/"))))]
        assert not leftovers, f"opponent observation term {name} still reads the robot: {leftovers}"
    return group


# ── Steering: walk at the other robot ─────────────────────────────────────────


def approach_command(me, other, speed: float, turn_gain: float = 2.0,
                     max_turn: float = 0.75) -> torch.Tensor:
    """`[vx, vy, wz]` in `me`'s body frame that turns it to face `other` and walks.

    Forward speed falls off with the heading error (cosine, clamped at 0), so a
    robot facing away turns on the spot before it walks.
    """
    d = other.data.root_link_pos_w[:, :2] - me.data.root_link_pos_w[:, :2]
    w, x, y, z = me.data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    err = torch.remainder(torch.atan2(d[:, 1], d[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
    vx = speed * torch.clamp(torch.cos(err), min=0.0)
    wz = torch.clamp(turn_gain * err, -max_turn, max_turn)
    return torch.stack((vx, torch.zeros_like(vx), wz), dim=-1)


# ── The action term ───────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class OpponentPolicyActionCfg(JointPositionActionCfg):
    """The opponent's joints, driven by a frozen policy (see the module docstring)."""

    checkpoint: str = ""
    observation_group: object = None
    #: The HeldCommand terms this entity's policy is shown.
    twist_name: str = TWIST_OPP
    posture_name: str = POSTURE_OPP
    #: Forward speed the opponent walks at the robot with, m/s (0 stands still).
    approach_speed: float = 0.3
    #: If set, each environment draws its opponent's speed from this range at
    #: every reset instead.
    approach_speed_range: tuple[float, float] | None = None
    neutral_height: float = 0.10647
    extra: dict = field(default_factory=dict)

    def build(self, env: "ManagerBasedRlEnv") -> "OpponentPolicyAction":
        return OpponentPolicyAction(self, env)


class OpponentPolicyAction(JointPositionAction):
    cfg: OpponentPolicyActionCfg

    def __init__(self, cfg: OpponentPolicyActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self._hidden = True  # from here on the manager sees 0 dims
        paths = [p.strip() for p in cfg.checkpoint.split(",") if p.strip()]
        self._actors = [FrozenActor(Path(p), str(env.device)) for p in paths]
        self._actor = self._actors[0]
        dims = {a.obs_dim for a in self._actors}
        assert len(dims) == 1, f"opponent pool mixes observation widths {sorted(dims)}"
        self._which = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._speed = torch.full((self.num_envs,), cfg.approach_speed, device=self.device)
        self._draw(torch.arange(self.num_envs, device=self.device))
        self._obs_manager = None

    def _draw(self, env_ids: torch.Tensor) -> None:
        """A pool member and an approach speed for these environments."""
        n = len(env_ids)
        if len(self._actors) > 1:
            self._which[env_ids] = torch.randint(len(self._actors), (n,), device=self.device)
        if self.cfg.approach_speed_range is not None:
            lo, hi = self.cfg.approach_speed_range
            self._speed[env_ids] = torch.empty(n, device=self.device).uniform_(lo, hi)

    def _act(self, obs: torch.Tensor) -> torch.Tensor:
        """Each environment's own pool member's action."""
        if len(self._actors) == 1:
            return self._actor(obs)
        out = torch.stack([a(obs) for a in self._actors])          # [P, N, A]
        return out.gather(0, self._which[None, :, None].expand(1, *out.shape[1:]))[0]

    @property
    def action_dim(self) -> int:
        return 0 if getattr(self, "_hidden", False) else self._action_dim

    def _observe(self) -> torch.Tensor:
        if self._obs_manager is None:
            from mjlab.managers.observation_manager import ObservationManager

            self._obs_manager = ObservationManager(
                {self.cfg.entity_name: self.cfg.observation_group}, self._env)
        obs = self._obs_manager.compute_group(self.cfg.entity_name, update_history=True)
        assert obs.shape[-1] == self._actor.obs_dim, (
            f"opponent observation is {obs.shape[-1]} wide, its policy takes {self._actor.obs_dim}")
        return obs

    def commands(self) -> tuple[torch.Tensor, torch.Tensor]:
        """`(twist [N,3], posture [N,4])` to show the policy this step.

        The opponent: shikiri until the tachiai, then walk at the robot.
        """
        from .mdp import SHIKIRI_POSTURE, bout, shikiri_or  # mdp imports this module

        env = self._env
        state = bout(env)
        state.update()
        walk = approach_command(self._entity, env.scene["robot"], self._speed)
        neutral = torch.tensor([0.0, 0.0, 0.0, self.cfg.neutral_height], device=self.device)
        crouch = neutral + torch.tensor(SHIKIRI_POSTURE, device=self.device)
        from .mdp import hold_mark

        return (shikiri_or(walk, state.started, hold_mark(env, self.cfg.entity_name)),
                shikiri_or(neutral.expand(self.num_envs, 4), state.started, crouch))

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions  # the learner sends none
        twist, posture = self.commands()
        self._env.command_manager.get_term(self.cfg.twist_name).set(twist)
        self._env.command_manager.get_term(self.cfg.posture_name).set(posture)
        super().process_actions(self._act(self._observe()))
        self._overlay_arms()

    def _overlay_arms(self) -> None:
        """Arm poses over the policy's targets: the raised-hands shikiri (`arms.py`)."""
        from .arms import shikiri_arms

        shikiri_arms(self, self.cfg.entity_name)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self._obs_manager is not None:
            self._obs_manager.reset(env_ids)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._draw(env_ids)
