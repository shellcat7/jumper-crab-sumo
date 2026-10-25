"""jumper.sumo.gyoji_policy: the gyoji's action, observations, rewards and terminations.

What a gyoji's movements are for (the International Sumo Federation's refereeing
regulations, Arts. 31-34; the Japan Sumo Association's practice is the same):

- be "in a position, angle, or form where a decision can be clearly made" -- here,
  a side view of the two wrestlers: looking across the line they push along, so
  neither hides the other and the foot that crosses the straw first is seen;
- move "swift and nimble, in order not to obstruct the movements of the athletes":
  never in their way, never touched, quick when the bout turns and still when it
  does not;
- do not step on the straw, and do not turn one's back to the front of the ring
  (shomen, -y here: the chief judge and the camera) -- so watch from the far
  side, mukō-jōmen, when the bout allows;
- crouch close to the action at the decisive moment, to see the feet;
- at the decision, point to the winner (the gunbai, `arms.GYOJI_RAISE`), stand.

The scripted gyoji (`gyoji.py`) aims at the first of these with a fixed rule; on
the reels of 10/13 (`devlog/2026-10-14.md`) it had the side view only 38-56 % of
a bout, touched a wrestler 15 % of it in the busiest, was beyond the straw 10-19 %
and was always on the move. Here a policy chooses the gyoji's velocity and crouch
from what it sees, paid for each of those aims; its legs are the same frozen
walking policy the scripted gyoji uses, so what is learned is where to be.

The action, four numbers each clipped to [-1, 1]:

    vx, vy, wz    body-frame velocity: +-0.35 m/s, +-0.30 m/s, +-0.75 rad/s, with a
                  dead band (`DEADBAND`) so that "nothing" is exactly nothing, and
                  the gyoji stands still -- its legs held at the stance, as the
                  scripted one's -- instead of stepping in place
    crouch        0 or less: upright; 1: the body 1.2 cm lower, pitched 0.08 rad to
                  look down (standing still, the low stance, `arms.STANCE_LOW`)

Through the shikiri it is not asked: it stands where it was put, as a gyoji does
with the gunbai up while the wrestlers settle.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from ..gyoji import GYOJI, GyojiPolicyAction, GyojiPolicyActionCfg
from ..opponent import OPPONENT, FrozenActor, OpponentPolicyAction
from ..tactics.mdp import _to_body, _yaw, opponent_relative, own_motion

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

GYOJI_ACTION_DIM = 4
#: Knobs for a livelier gyoji (10/14 afternoon, the user: its moves were still dull --
#: it followed the bout late and slowly, 0.16 m/s against the scripted one's 0.25).
#: A trained policy must be replayed with the knobs it was trained with: the speeds
#: and the smoothing are what its outputs mean.
#:   JUMPER_GYOJI_SPEED "vx,vy,wz"   at an output of 1 (default 0.35,0.30,0.75)
#:   JUMPER_GYOJI_SMOOTH_S           how fast commands follow the outputs (0.10 s)
#:   JUMPER_GYOJI_LEAD_S             the watching rewards score where the wrestlers
#:                                   will be this far ahead (0: where they are)
#:   JUMPER_GYOJI_SIDE_POWER         side view paid as |sin|^p (2): higher pays
#:                                   only a view close to square-on
SPEED = tuple(float(v) for v in os.environ.get("JUMPER_GYOJI_SPEED", "0.35,0.30,0.75").split(","))
DEADBAND = 0.10
SMOOTHING_S = float(os.environ.get("JUMPER_GYOJI_SMOOTH_S", "0.10"))
LEAD_S = float(os.environ.get("JUMPER_GYOJI_LEAD_S", "0"))
SIDE_POWER = float(os.environ.get("JUMPER_GYOJI_SIDE_POWER", "2"))
#: JUMPER_GYOJI_CROUCH_ENGAGED=1: the crouch is also paid (half) while the wrestlers
#: are locked together, not only at the straw.
CROUCH_ENGAGED = os.environ.get("JUMPER_GYOJI_CROUCH_ENGAGED", "") == "1"
CROUCH_HEIGHT = 0.012            # m lower at a full crouch, walking (the shikiri's)
CROUCH_PITCH = 0.08              # rad, nose down
STILL_V, STILL_W = 0.02, 0.05    # m/s, rad/s: a command this small, asked for nothing, is nothing
#: Where it watches from: this far from the wrestlers' midpoint (about 1.6 m on a
#: real dohyo), and no nearer either wrestler's base than `CLEAR` (claws and
#: gunbai reach about 0.2 m each).
WATCH_DIST = 0.55
CLEAR = 0.45
SHOMEN = (0.0, -1.0)             # the front of the ring (audience.py: shomen is -y)


# ── The action ────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class LearnedGyojiActionCfg(GyojiPolicyActionCfg):
    """The gyoji's legs (`checkpoint`, a frozen walking policy, as the scripted
    gyoji's) steered by a learned policy: the learner's own action in training;
    in a replay, `policy` (a trained checkpoint) on `policy_group`."""

    policy: str = ""
    policy_group: object = None

    def build(self, env: "ManagerBasedRlEnv") -> "LearnedGyojiAction":
        return LearnedGyojiAction(self, env)


class LearnedGyojiAction(GyojiPolicyAction):
    cfg: LearnedGyojiActionCfg

    def __init__(self, cfg: LearnedGyojiActionCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        n, dev = self.num_envs, self.device
        self._policy = FrozenActor(Path(cfg.policy), str(dev)) if cfg.policy else None
        self._policy_obs = None
        #: The policy's last output, unclipped (its `actions` input), and clipped.
        self._raw_choice = torch.zeros(n, GYOJI_ACTION_DIM, device=dev)
        self._choice = torch.zeros(n, GYOJI_ACTION_DIM, device=dev)
        self._cmd = torch.zeros(n, 3, device=dev)
        self._crouch = torch.zeros(n, device=dev)
        self._still = torch.ones(n, dtype=torch.bool, device=dev)
        self._stance_w = torch.ones(n, device=dev)
        self._speed_scale = torch.tensor(SPEED, device=dev)
        self._alpha = min(1.0, env.step_dt / SMOOTHING_S)

    @property
    def action_dim(self) -> int:
        if not getattr(self, "_hidden", False):
            return self._action_dim          # the joint path's 20
        return 0 if self.cfg.policy else GYOJI_ACTION_DIM

    def _observe_policy(self) -> torch.Tensor:
        if self._policy_obs is None:
            from mjlab.managers.observation_manager import ObservationManager

            self._policy_obs = ObservationManager({"gyoji_policy": self.cfg.policy_group}, self._env)
        obs = self._policy_obs.compute_group("gyoji_policy", update_history=True)
        assert obs.shape[-1] == self._policy.obs_dim, (
            f"gyoji observation is {obs.shape[-1]} wide, its policy takes {self._policy.obs_dim}")
        return obs

    def commands(self) -> tuple[torch.Tensor, torch.Tensor]:
        from ..mdp import bout
        from ..referee import referee

        env = self._env
        referee(env, self.cfg.radius).update()
        waiting = ~bout(env).started
        a = self._choice[:, :3]
        a = torch.sign(a) * (a.abs() - DEADBAND).clamp(min=0.0) / (1.0 - DEADBAND)
        target = torch.where(waiting[:, None], torch.zeros_like(a), a * self._speed_scale)
        self._cmd = self._cmd + self._alpha * (target - self._cmd)
        settle = ((target == 0).all(-1) & (torch.linalg.norm(self._cmd[:, :2], dim=-1) < STILL_V)
                  & (self._cmd[:, 2].abs() < STILL_W))
        self._cmd = torch.where(settle[:, None], torch.zeros_like(self._cmd), self._cmd)
        self._still = settle
        crouch = torch.where(waiting, torch.zeros_like(self._crouch), self._choice[:, 3].clamp(0.0, 1.0))
        self._crouch = self._crouch + self._alpha * (crouch - self._crouch)
        zero = torch.zeros_like(self._crouch)
        posture = torch.stack((zero, CROUCH_PITCH * self._crouch, zero,
                               self.cfg.neutral_height - CROUCH_HEIGHT * self._crouch), -1)
        return self._cmd.clone(), posture

    def process_actions(self, actions: torch.Tensor) -> None:
        raw = self._policy(self._observe_policy()) if self._policy is not None else actions
        self._raw_choice = raw.clone()
        self._choice = raw.clamp(-1.0, 1.0)
        # OpponentPolicyAction's: commands(), the walking policy, the arms.
        OpponentPolicyAction.process_actions(self, actions)

    def _overlay_arms(self) -> None:
        """The gunbai as the scripted gyoji holds it; standing still, the legs held
        at the stance (eased over about 50 ms), as low as the crouch asks."""
        from ..arms import GYOJI_HOLD, GYOJI_RAISE, STANCE, STANCE_LOW, overlay
        from ..referee import referee

        ones = torch.ones(self.num_envs, device=self.device)
        self._stance_w = self._stance_w + 0.1 * (self._still.float() - self._stance_w)
        overlay(self, STANCE, self._stance_w)
        overlay(self, STANCE_LOW, self._stance_w * self._crouch)
        overlay(self, GYOJI_HOLD, ones)
        overlay(self, GYOJI_RAISE, referee(self._env, self.cfg.radius).decided.float())

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for t in (self._raw_choice, self._choice, self._cmd):
            t[ids] = 0.0
        self._crouch[ids] = 0.0
        self._still[ids] = True
        self._stance_w[ids] = 1.0
        if self._policy_obs is not None:
            self._policy_obs.reset(env_ids)


# ── Geometry of the bout, seen by the gyoji ───────────────────────────────────


def _bout_geometry(env: "ManagerBasedRlEnv", lead: float = 0.0) -> dict[str, torch.Tensor]:
    """Ring-centred positions and the pair's line. Not cached by step: the
    observation after a reset is computed on the same step as the rewards before
    it, with the bodies moved in between. `lead` (s): the wrestlers where their
    velocities will have taken them (the watching rewards, `LEAD_S`)."""
    origin = env.scene.env_origins[:, :2]
    ra, rb = env.scene["robot"].data, env.scene[OPPONENT].data
    a = ra.root_link_pos_w[:, :2] - origin + lead * ra.root_link_lin_vel_w[:, :2]
    b = rb.root_link_pos_w[:, :2] - origin + lead * rb.root_link_lin_vel_w[:, :2]
    g = env.scene[GYOJI].data.root_link_pos_w[:, :2] - origin
    mid = (a + b) / 2
    axis = b - a
    sep = torch.linalg.norm(axis, dim=-1).clamp(min=1e-3)
    u = axis / sep[:, None]
    rel = g - mid
    dist = torch.linalg.norm(rel, dim=-1).clamp(min=1e-3)
    across = u[:, 0] * rel[:, 1] - u[:, 1] * rel[:, 0]
    return {"a": a, "b": b, "g": g, "mid": mid, "u": u, "sep": sep, "rel": rel,
            "dist": dist, "side": across.abs() / dist}


def _phase(env: "ManagerBasedRlEnv", radius: float):
    """(waiting, live, decided) masks."""
    from ..mdp import bout
    from ..referee import referee

    r = referee(env, radius)
    r.update()
    started = bout(env).started
    return ~started, started & ~r.decided, r.decided


# ── Observations ──────────────────────────────────────────────────────────────


def pair_geometry(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """The two wrestlers as a pair, in the gyoji's frame (5): their midpoint (x, y),
    the line between them as (cos 2phi, sin 2phi) -- a line, not a direction, so
    east and west may swap -- and their separation."""
    geo = _bout_geometry(env)
    yaw = _yaw(env.scene[GYOJI].data.root_link_quat_w)
    mid = _to_body(geo["mid"] - geo["g"], yaw)
    u = _to_body(geo["u"], yaw)
    c2, s2 = u[:, 0] ** 2 - u[:, 1] ** 2, 2 * u[:, 0] * u[:, 1]
    return torch.cat((mid, c2[:, None], s2[:, None], geo["sep"][:, None]), -1)


def gyoji_ring(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """The ring seen from the gyoji (7): the way to the centre (x, y), its own and
    each wrestler's distance from the centre over the radius, and the direction
    of the front of the ring (shomen) in its frame."""
    geo = _bout_geometry(env)
    yaw = _yaw(env.scene[GYOJI].data.root_link_quat_w)
    to_centre = _to_body(-geo["g"], yaw)
    front = _to_body(torch.tensor(SHOMEN, device=env.device).expand_as(geo["g"]), yaw)
    r = torch.stack([torch.linalg.norm(geo[k], dim=-1) for k in ("g", "a", "b")], -1) / radius
    return torch.cat((to_centre, r, front), -1)


def gyoji_bout(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """The bout as the gyoji runs it (6): started, seconds since the tachiai / 10,
    decided, won by east, won by west, the wrestlers touching."""
    from ..mdp import bout
    from ..referee import _found, referee

    state = bout(env)
    r = referee(env, radius)
    r.update()
    since = torch.where(state.started,
                        (env.common_step_counter - state.start_step).float() * env.step_dt / 10.0,
                        torch.zeros_like(state.elapsed))
    touching = _found(env, "bout_contact")
    return torch.stack((state.started.float(), since.clamp(max=2.0), r.decided.float(),
                        (r.decided & (r.winner == 0)).float(), (r.decided & (r.winner == 1)).float(),
                        touching.float()), -1)


def gyoji_choice(env: "ManagerBasedRlEnv", action_name: str = GYOJI) -> torch.Tensor:
    """The gyoji policy's last output (4), unclipped -- the learner's and a frozen
    policy's alike."""
    return env.action_manager.get_term(action_name)._raw_choice


def gyoji_touched(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """1 while the gyoji touches either wrestler (critic; and a cost)."""
    from ..referee import _found

    return (_found(env, "gyoji_touch_east") | _found(env, "gyoji_touch_west")).float()[:, None]


def gyoji_policy_terms(radius: float) -> dict:
    """The gyoji policy's observation (44), in a fixed order."""
    return {
        "own_motion": ObservationTermCfg(func=own_motion, params={"me": GYOJI}),
        "east": ObservationTermCfg(func=opponent_relative, params={"me": GYOJI, "other": "robot"}),
        "west": ObservationTermCfg(func=opponent_relative, params={"me": GYOJI, "other": OPPONENT}),
        "pair": ObservationTermCfg(func=pair_geometry),
        "ring": ObservationTermCfg(func=gyoji_ring, params={"radius": radius}),
        "bout": ObservationTermCfg(func=gyoji_bout, params={"radius": radius}),
        "actions": ObservationTermCfg(func=gyoji_choice),
    }


def gyoji_policy_group(radius: float) -> ObservationGroupCfg:
    return ObservationGroupCfg(terms=gyoji_policy_terms(radius))


# ── Rewards (per second; the reward manager multiplies by the step) ──────────


def watch_side(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """In the bout: how square-on the gyoji looks across the pair's line, sin^2
    of the angle -- 1 for a side view, 0 looking down the line (one hides the
    other, and which foot crosses first cannot be seen)."""
    _, live, _ = _phase(env, radius)
    return torch.where(live, _bout_geometry(env, LEAD_S)["side"] ** SIDE_POWER, torch.zeros(env.num_envs, device=env.device))


def watch_distance(env: "ManagerBasedRlEnv", radius: float, dist: float = WATCH_DIST,
                   width: float = 0.12) -> torch.Tensor:
    """In the bout: near the watching distance from the pair's midpoint."""
    _, live, _ = _phase(env, radius)
    d = _bout_geometry(env, LEAD_S)["dist"]
    return torch.where(live, torch.exp(-((d - dist) / width).square()), torch.zeros_like(d))


def facing(env: "ManagerBasedRlEnv", radius: float, width: float = 0.35) -> torch.Tensor:
    """Facing the bout (the pair's midpoint); after the decision, the winner."""
    from ..referee import referee

    geo = _bout_geometry(env, LEAD_S)
    r = referee(env, radius)
    winner = torch.where((r.winner == 0)[:, None], geo["a"], geo["b"])
    look = torch.where((r.decided & (r.winner >= 0))[:, None], winner, geo["mid"])
    d = look - geo["g"]
    yaw = _yaw(env.scene[GYOJI].data.root_link_quat_w)
    err = torch.remainder(torch.atan2(d[:, 1], d[:, 0]) - yaw + math.pi, 2 * math.pi) - math.pi
    waiting, _, _ = _phase(env, radius)
    return torch.where(waiting, torch.zeros_like(err), torch.exp(-(err / width).square()))


def back_to_front(env: "ManagerBasedRlEnv", radius: float, scale: float = 0.15) -> torch.Tensor:
    """In the bout: +1 watching from the far side (mukō-jōmen, facing the front),
    -1 from the front's side (its back to shomen); smooth across the pair."""
    _, live, _ = _phase(env, radius)
    rel = _bout_geometry(env, LEAD_S)["rel"]
    far = -(rel[:, 0] * SHOMEN[0] + rel[:, 1] * SHOMEN[1])
    return torch.where(live, torch.tanh(far / scale), torch.zeros_like(far))


def too_close(env: "ManagerBasedRlEnv", clear: float = CLEAR, width: float = 0.10) -> torch.Tensor:
    """Nearer a wrestler's base than `clear`: ((clear - d) / width)^2, both summed."""
    geo = _bout_geometry(env)
    cost = torch.zeros(env.num_envs, device=env.device)
    for k in ("a", "b"):
        d = torch.linalg.norm(geo["g"] - geo[k], dim=-1)
        cost = cost + ((clear - d) / width).clamp(min=0.0).square()
    return cost


def touching(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return gyoji_touched(env)[:, 0]


def hidden_from_front(env: "ManagerBasedRlEnv", radius: float, width: float = 0.15,
                      depth: float = 0.05) -> torch.Tensor:
    """In the bout: a wrestler between the gyoji and the front, as the front's camera
    sees it (10/15: from the far side the gyoji stood straight behind the pair and the
    pair hid it). Per wrestler, how much it is in line with the gyoji across the front
    (exp(-(lateral / width)^2)) times whether it is nearer the front (a sigmoid of how
    far nearer, over `depth`), summed: 0 in full view, up to 2 behind both."""
    _, live, _ = _phase(env, radius)
    geo = _bout_geometry(env)
    front = torch.tensor(SHOMEN, device=env.device)
    across = torch.stack((-front[1], front[0]))
    cost = torch.zeros(env.num_envs, device=env.device)
    for k in ("a", "b"):
        d = geo[k] - geo["g"]
        ahead = d @ front
        lateral = d @ across
        cost = cost + torch.exp(-(lateral / width).square()) * torch.sigmoid(ahead / depth)
    return torch.where(live, cost, torch.zeros_like(cost))


def saw_decision(env: "ManagerBasedRlEnv", radius: float, body: float = 0.13) -> torch.Tensor:
    """At the decision (10/20: a gyoji is there to see it): 1/step_dt on the step the
    referee calls the bout, times how well the gyoji saw the loser at that moment -- 0 if
    the winner stood between them (its base within `body` of the gyoji's line to the
    loser, and nearer than the loser), else 0.5 + 0.5 x sin^2 of the angle between that
    line and the loser's way out (a foot crossing the straw is seen from the side of its
    path, not from in front of or behind it). The weight is then what one decision pays."""
    from ..referee import referee

    r = referee(env, radius)
    r.update()
    now = r.decided & (r.decided_at == r.time) & (r.winner >= 0)
    geo = _bout_geometry(env)
    lose_a = (r.winner == 1)[:, None]
    loser = torch.where(lose_a, geo["a"], geo["b"])
    winner = torch.where(lose_a, geo["b"], geo["a"])
    sight = loser - geo["g"]
    L = torch.linalg.norm(sight, dim=-1).clamp(min=1e-3)
    u = sight / L[:, None]
    w = winner - geo["g"]
    along = (w * u).sum(-1)
    lateral = (u[:, 0] * w[:, 1] - u[:, 1] * w[:, 0]).abs()
    blocked = (lateral < body) & (along > 0) & (along < L)
    out = loser / torch.linalg.norm(loser, dim=-1, keepdim=True).clamp(min=1e-3)
    side = (u[:, 0] * out[:, 1] - u[:, 1] * out[:, 0]).square()
    seen = torch.where(blocked, torch.zeros_like(side), 0.5 + 0.5 * side)
    return torch.where(now, seen / env.step_dt, torch.zeros_like(seen))


def beyond_straw(env: "ManagerBasedRlEnv", start: float = 0.72, width: float = 0.10) -> torch.Tensor:
    """Out towards the straw (at 0.8 m) and beyond: ((r - start) / width)^2."""
    r = torch.linalg.norm(_bout_geometry(env)["g"], dim=-1)
    return ((r - start) / width).clamp(min=0.0).square()


def moving(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """The gyoji's speed over the clay, squared (m/s)^2: quick when needed only."""
    return env.scene[GYOJI].data.root_link_lin_vel_w[:, :2].square().sum(-1)


def turning(env: "ManagerBasedRlEnv") -> torch.Tensor:
    return env.scene[GYOJI].data.root_link_ang_vel_b[:, 2].square()


def bobbing(env: "ManagerBasedRlEnv") -> torch.Tensor:
    """Rocking and bobbing of the gyoji's body: roll and pitch rates squared, and
    the vertical speed (x10) squared."""
    d = env.scene[GYOJI].data
    return d.root_link_ang_vel_b[:, :2].square().sum(-1) + (10.0 * d.root_link_lin_vel_w[:, 2]).square()


def crouch_at_the_straw(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """In the bout: the crouch, paid when a wrestler is near the straw (the
    decisive moment, the feet to be seen), a small cost elsewhere."""
    _, live, _ = _phase(env, radius)
    geo = _bout_geometry(env)
    edge = ((torch.maximum(torch.linalg.norm(geo["a"], dim=-1), torch.linalg.norm(geo["b"], dim=-1))
             / radius - 0.7) / 0.3).clamp(0.0, 1.0)
    crouch = env.action_manager.get_term(GYOJI)._crouch
    if CROUCH_ENGAGED:
        # A gyoji bends to watch a clinch, lower at the straw (10/14, the user: like
        # a real gyoji): half the pay while the two are locked together.
        from ..referee import _found

        edge = torch.maximum(edge, 0.5 * _found(env, "bout_contact").float())
    return torch.where(live, crouch * (edge - 0.2 * (1.0 - edge)), torch.zeros_like(edge))


def standing_after(env: "ManagerBasedRlEnv", radius: float) -> torch.Tensor:
    """After the decision: standing still (the call is made standing)."""
    _, _, decided = _phase(env, radius)
    v = env.scene[GYOJI].data.root_link_lin_vel_w[:, :2].square().sum(-1)
    return torch.where(decided, torch.exp(-v / 0.005), torch.zeros_like(v))


def gyoji_ended(env: "ManagerBasedRlEnv", terms: tuple[str, ...] = ("gyoji_fell", "gyoji_off")) -> torch.Tensor:
    tm = env.termination_manager
    fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in terms:
        fired |= tm.get_term(name).bool()
    return fired.float()


# ── Terminations ──────────────────────────────────────────────────────────────


def gyoji_fell(env: "ManagerBasedRlEnv", limit_angle: float = math.radians(50.0)) -> torch.Tensor:
    g = env.scene[GYOJI].data.projected_gravity_b
    return torch.acos(torch.clamp(-g[:, 2], -1.0, 1.0)) > limit_angle


def gyoji_off(env: "ManagerBasedRlEnv", limit: float = 1.05) -> torch.Tensor:
    """Off the top of the mound (its edge 1.18 m out, square)."""
    return _bout_geometry(env)["g"].abs().max(-1).values > limit
