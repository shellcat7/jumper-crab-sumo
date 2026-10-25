"""The gyoji's judgement in jumper.sumo (jumper-challenge-2026).

One `Referee` for all environments (kept on the env as `sumo_referee`), advanced
once per step. It watches the bout `BoutState` runs and does what a gyoji does, in
the terms a hexapod allows. With a gyoji in the ring (`officiates`) it also runs
the shikiri; without one (`JUMPER_SUMO_REFEREE=1`) it only judges, so an
evaluation keeps the training's start rules and gains the kimarite counts
(`tally`).

- **Shikiri.** "Miatte" at the reset, with both wrestlers' claws raised (`arms.py`);
  "te o tsuite" (hands down) as the first starts to lower them; "matta nashi"
  once both wrestlers' claws have been confirmed down and behind their shikiri
  lines (`BoutState.hands`, `BoutState.line_x`); "hakkeyoi" at the tachiai.
- **Matta.** A wrestler whose base drifts more than `MATTA_DRIFT` from where it
  settled is a false start, and so is a shikiri that reaches `MATTA_AFTER_S`
  without both pairs of claws down: "matta", and the shikiri restarts. With a
  gyoji the bout never starts without the hands confirmed (`BoutState`'s own
  fallback start is switched off), except after `MAX_MATTA` restarts in a row.
- **During the bout**, while the two are in contact: "nokotta" while either is
  moving, "hakkeyoi" to urge them on when both have stood still for `STALL_S`.
- **Shobu ari** when one is out -- a foot on the ground beyond the straw, the
  moment it touches (`mdp.stepped_out`) -- or down: anything but a foot on the
  clay (`mdp.touched_down`), or tipped past `FALL_ANGLE`. The same tests the
  bout's terminations use.
- **The winner**, east (the robot, the learner) or west (the opponent), and the
  **kimarite** read off the moment of decision (`classify`).

Kimarite, as a hexapod with two claws can show them. The Japan Sumo Association
recognises 82 techniques and 5 non-techniques; the ones below are those whose
defining feature survives the change of body, decided from what the simulator
knows at the decision -- contact between the two, whether their bodies were
pressed together, who is behind whom, and which way the loser was moving.

There is no belt to grip, so "yori" and "oshi" are told apart by how close the
two are: a wrestler who drives with its body against the other's (chest to
chest, base to base) is doing what yori is; one whose claws and legs alone are
on the other, bodies apart, is pushing -- oshi. (Claw contact as a "grip" was
tried first: a pushing crab always has a claw on its opponent, and the first
86 decisions were all yorikiri.)

    oshidashi / oshitaoshi    pushed out / down, bodies apart
    yorikiri / yoritaoshi     driven out / down, bodies pressed together
    tsukidashi / tsukitaoshi  thrust apart, out / down after the contact broke
    okuridashi / okuritaoshi  from behind
    hikiotoshi                the loser went down or out moving forwards, not
                              pushed (pulled or side-stepped)
    isamiashi                 non-technique: the loser stepped out by itself
    koshikudake               non-technique: the loser went down by itself
    dotai                     both at once: a re-match (torinaoshi)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

#: Index -> (romaji, kanji). Index 0 is "none yet".
KIMARITE = (
    ("", ""),
    ("oshidashi", "押し出し"), ("oshitaoshi", "押し倒し"),
    ("yorikiri", "寄り切り"), ("yoritaoshi", "寄り倒し"),
    ("tsukidashi", "突き出し"), ("tsukitaoshi", "突き倒し"),
    ("okuridashi", "送り出し"), ("okuritaoshi", "送り倒し"),
    ("hikiotoshi", "引き落とし"),
    ("isamiashi", "勇み足"), ("koshikudake", "腰砕け"),
    ("dotai", "同体"),
    # With claws (tactics/waza.py, 10/10 night): grip throws and slap-downs.
    ("uwatenage", "上手投げ"), ("shitatenage", "下手投げ"),
    ("hatakikomi", "はたき込み"), ("tsukiotoshi", "突き落とし"),
    # A crab's holds: a leg picked (ashitori), an arm pulled (tottari).
    ("ashitori", "足取り"), ("tottari", "とったり"),
)
_K = {name: i for i, (name, _) in enumerate(KIMARITE)}
EAST, WEST = 0, 1
SIDE_NAME = ("東", "西")

MATTA_DRIFT = 0.05          # m a wrestler may drift during shikiri
MATTA_AFTER_S = 4.0         # a shikiri this long without the hands confirmed is a matta
MAX_MATTA = 3               # restarts in a row before the bout is started regardless
HANDS_CALL_S = 0.5          # "te o tsuite" no earlier than this into each shikiri
FALL_ANGLE = 0.8726646      # 50 deg, posture's fell_over
TOUCH_RECENT_S = 0.25       # contact this recent counts as "still pushing"
TOUCH_EARLIER_S = 1.5       # ... this recent as "thrust, then apart"
MOVING = 0.06               # m/s: a wrestler faster than this is working
STALL_S = 1.5               # both slower than MOVING this long, in contact: stalled
NOKOTTA_EVERY_S = 0.8
#: A called bout stays on screen this long before the ring resets (gyoji replays):
#: the wrestlers stopped (arms.STOP_S) and the gyoji turned to the winner. 1.2 s
#: until 10/10 night, when they still went on pushing through it.
SHOW_S = 2.0


def _found(env, name: str) -> torch.Tensor:
    found = env.scene.sensors[name].data.found
    return found.reshape(found.shape[0], -1).sum(-1) > 0


def _tilt(q: torch.Tensor) -> torch.Tensor:
    """Angle of the body's z axis from vertical, from a [N,4] wxyz quaternion."""
    w, x, y, z = q.unbind(-1)
    return torch.acos(torch.clamp(1 - 2 * (x * x + y * y), -1.0, 1.0))


def _yaw(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class Referee:
    def __init__(self, env: "ManagerBasedRlEnv", radius: float):
        n, dev = env.num_envs, env.device
        self.env, self.radius = env, radius
        self.decided = torch.zeros(n, dtype=torch.bool, device=dev)
        self.winner = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.kimarite = torch.zeros(n, dtype=torch.long, device=dev)
        self.decided_at = torch.zeros(n, device=dev)      # bout time of the decision
        self.time = torch.zeros(n, device=dev)            # time since the bout's reset
        self.since_touch = torch.full((n,), 1e3, device=dev)
        self.since_chest = torch.full((n,), 1e3, device=dev)    # bodies pressed together
        self.settled = torch.zeros(n, 2, 2, device=dev)   # shikiri base xy, per side
        self.has_settled = torch.zeros(n, dtype=torch.bool, device=dev)
        #: True when the gyoji runs the shikiri (matta, no start without hands).
        self.officiates = False
        #: Decisions so far, every environment: [kimarite, east won / west won / dotai].
        self.tally = torch.zeros(len(KIMARITE), 3, dtype=torch.long, device=dev)
        self.mattas = torch.zeros(n, dtype=torch.long, device=dev)   # in a row, this bout
        self.still = torch.zeros(n, device=dev)           # time both have been still in contact
        #: Environment 0's calls, for overlays: (bout time, text).
        self.calls: list[tuple[float, str]] = []
        self._last_call = -1e9
        self._env0 = {"hands": False, "confirmed": False}
        self._step: int | None = None

    # -- bookkeeping ---------------------------------------------------------
    def reset(self, env_ids: torch.Tensor) -> None:
        for t in (self.decided, self.has_settled):
            t[env_ids] = False
        self.winner[env_ids] = -1
        self.kimarite[env_ids] = 0
        self.time[env_ids] = 0.0
        self.decided_at[env_ids] = 0.0
        self.since_touch[env_ids] = 1e3
        self.since_chest[env_ids] = 1e3
        self.mattas[env_ids] = 0
        self.still[env_ids] = 0.0
        if bool((env_ids == 0).any()):
            self.calls.append((0.0, "見合って"))
            self._last_call = -1e9
            self._env0 = {"hands": False, "confirmed": False}

    def _call(self, text: str) -> None:
        self.calls.append((float(self.time[0]), text))

    # -- the step ------------------------------------------------------------
    def update(self) -> None:
        env = self.env
        step = int(env.common_step_counter)
        if step == self._step:
            return
        self._step = step
        from .mdp import bout

        state = bout(env)
        state.update()
        dt = env.step_dt
        self.time += dt
        east, west = env.scene["robot"].data, env.scene["opponent"].data
        origin = env.scene.env_origins[:, :2]
        pos = torch.stack((east.root_link_pos_w[:, :2], west.root_link_pos_w[:, :2]), 1) - origin[:, None]

        touch = _found(env, "bout_contact")
        chest = _found(env, "chest_contact")
        self.since_touch = torch.where(touch, torch.zeros_like(self.since_touch), self.since_touch + dt)
        self.since_chest = torch.where(chest, torch.zeros_like(self.since_chest), self.since_chest + dt)

        # Shikiri: remember where each settled; drifting off it is a matta, and so
        # is a shikiri that runs long without both pairs of claws down.
        waiting = ~state.started
        settle = waiting & ~self.has_settled & (state.hands > 0)
        self.settled[settle] = pos[settle]
        self.has_settled |= settle
        drift = torch.linalg.norm(pos - self.settled, dim=-1).max(-1).values
        matta = waiting & ((self.has_settled & (drift > MATTA_DRIFT))
                           | (state.elapsed >= MATTA_AFTER_S + getattr(state, "ceremony", 0.0))) & self.officiates
        if bool(matta.any()):
            ids = matta.nonzero().flatten()
            self.mattas[ids] += 1
            force = ids[self.mattas[ids] > MAX_MATTA]
            again = ids[self.mattas[ids] <= MAX_MATTA]
            state.elapsed[again] = 0.0
            state.hands[again] = 0.0
            self.has_settled[again] = False
            state.started[force] = True
            state.start_step[force] = step
            if bool((again == 0).any()):
                self._call("待った")
                self._env0 = {"hands": False, "confirmed": False}
        if bool(waiting[0]) and not bool(matta[0]):
            # "Te o tsuite" as the first of the two starts to bring its claws down.
            first = max(HANDS_CALL_S, float(state.lower_at[0].min()))
            if not self._env0["hands"] and float(state.elapsed[0]) >= first:
                self._call("手をついて")
                self._env0["hands"] = True
            # "Matta nashi" as the second of the two starts to bring its claws down:
            # from here the tachiai comes the moment both pairs are down.
            if (self._env0["hands"] and not self._env0["confirmed"]
                    and float(state.elapsed[0]) >= float(state.lower_at[0].max())):
                self._call("待ったなし")
                self._env0["confirmed"] = True
        if bool(state.start_step[0] == step):
            if not self._env0["confirmed"]:
                # Hands confirmed on the very step the bout began.
                self._call("待ったなし")
                self._env0["confirmed"] = True
            self._call("はっけよい")
            self._last_call = float(self.time[0])

        # The decision.
        live = state.started & ~self.decided
        radial = torch.linalg.norm(pos, dim=-1)
        from .mdp import stepped_out, straw_edge, touched_down

        edge = straw_edge(self.radius)
        out = torch.stack((stepped_out(env, "robot", edge), stepped_out(env, "opponent", edge)), -1)
        tilted = torch.stack((_tilt(east.root_link_quat_w), _tilt(west.root_link_quat_w)), -1) > FALL_ANGLE
        down = tilted | torch.stack((touched_down(env, "robot"), touched_down(env, "opponent")), -1)
        lost = out | down                                  # [N, 2]: east lost, west lost
        decide = live & lost.any(-1)
        if bool(decide.any()):
            self._decide(decide, lost, out, pos)
        # In the bout: "nokotta" while they work, "hakkeyoi" when they have stalled.
        speed = torch.stack((torch.linalg.norm(east.root_link_lin_vel_w[:, :2], dim=-1),
                             torch.linalg.norm(west.root_link_lin_vel_w[:, :2], dim=-1)), -1)
        moving = (speed > MOVING).any(-1)
        locked = live & (self.since_touch < TOUCH_RECENT_S)
        self.still = torch.where(locked & ~moving, self.still + dt, torch.zeros_like(self.still))
        if bool(live[0]) and not bool(decide[0]) and bool(locked[0]):
            since = float(self.time[0]) - self._last_call
            if bool(moving[0]) and since > NOKOTTA_EVERY_S:
                self._call("のこった")
                self._last_call = float(self.time[0])
            elif float(self.still[0]) >= STALL_S and since > STALL_S:
                self._call("はっけよい")
                self._last_call = float(self.time[0])

    def _decide(self, decide, lost, out, pos) -> None:
        env = self.env
        both = lost.all(-1)
        loser = torch.where(lost[:, 0], 0, 1)
        winner = 1 - loser
        rows = torch.arange(env.num_envs, device=env.device)
        # The loser's velocity along its own heading: moving forwards = it was not pushed.
        datas = (env.scene["robot"].data, env.scene["opponent"].data)
        vel = torch.stack((datas[0].root_link_lin_vel_b[:, 0], datas[1].root_link_lin_vel_b[:, 0]), -1)
        yaw = torch.stack((_yaw(datas[0].root_link_quat_w), _yaw(datas[1].root_link_quat_w)), -1)
        loser_fwd = vel[rows, loser] > 0.05
        # Winner behind the loser: bearing from the loser's heading to the winner.
        rel = pos[rows, winner] - pos[rows, loser]
        bearing = torch.remainder(torch.atan2(rel[:, 1], rel[:, 0]) - yaw[rows, loser] + math.pi,
                                  2 * math.pi) - math.pi
        behind = bearing.abs() > 2.1                       # beyond about 120 deg
        fell = ~out[rows, loser]                           # down inside the ring
        touching = self.since_touch < TOUCH_RECENT_S
        thrust = (self.since_touch >= TOUCH_RECENT_S) & (self.since_touch < TOUCH_EARLIER_S)
        close = self.since_chest < TOUCH_RECENT_S

        k = torch.full_like(loser, _K["isamiashi"])
        k = torch.where(fell, _K["koshikudake"], k)
        k = torch.where(loser_fwd & (touching | thrust), _K["hikiotoshi"], k)
        k = torch.where(~loser_fwd & thrust, torch.where(fell, _K["tsukitaoshi"], _K["tsukidashi"]), k)
        k = torch.where(~loser_fwd & touching, torch.where(fell, _K["oshitaoshi"], _K["oshidashi"]), k)
        k = torch.where(~loser_fwd & touching & close,
                        torch.where(fell, _K["yoritaoshi"], _K["yorikiri"]), k)
        k = torch.where(behind & (touching | thrust), torch.where(fell, _K["okuritaoshi"], _K["okuridashi"]), k)
        from .tactics.waza import kimarite_hint

        k = kimarite_hint(env, rows, winner, loser, out[rows, loser], fell, k, _K)
        k = torch.where(both, _K["dotai"], k)

        column = torch.where(both, 2, winner)
        self.tally.index_put_((k[decide], column[decide]),
                              torch.ones_like(k[decide]), accumulate=True)
        self.decided |= decide
        self.winner = torch.where(decide, torch.where(both, -1, winner), self.winner)
        self.kimarite = torch.where(decide, k, self.kimarite)
        self.decided_at = torch.where(decide, self.time, self.decided_at)
        if bool(decide[0]):
            kanji = KIMARITE[int(k[0])][1]
            if bool(both[0]):
                self._call("同体　取り直し")
            else:
                self._call(f"勝負あり　{SIDE_NAME[int(winner[0])]}の勝ち　{kanji}")


def tally_text(r: Referee) -> str:
    """`oshidashi E12 W3; ...` for the kimarite decided so far (logs, CSV)."""
    parts = []
    for i, (name, _) in enumerate(KIMARITE):
        e, w, d = (int(v) for v in r.tally[i].tolist())
        if e or w or d:
            parts.append(f"{name} E{e} W{w}" + (f" D{d}" if d else ""))
    return "; ".join(parts)


def referee_watch(env: "ManagerBasedRlEnv", radius: float = 0.8) -> torch.Tensor:
    """Termination term that never fires: it only advances the referee every step
    (`JUMPER_SUMO_REFEREE=1`, where no gyoji's action term does)."""
    r = referee(env, radius)
    r.update()
    return torch.zeros_like(r.decided)


def referee(env: "ManagerBasedRlEnv", radius: float = 0.8) -> Referee:
    r = getattr(env, "sumo_referee", None)
    if r is None:
        r = Referee(env, radius)
        env.sumo_referee = r
    return r


def reset_referee(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None, radius: float = 0.8) -> None:
    """Event term (mode `reset`)."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    referee(env, radius).reset(env_ids)


def bout_over(env: "ManagerBasedRlEnv", radius: float = 0.8, show_s: float = SHOW_S) -> torch.Tensor:
    """Termination for replays with a gyoji: `show_s` after the decision, so the
    winner can be announced before the ring resets."""
    r = referee(env, radius)
    r.update()
    return r.decided & (r.time - r.decided_at >= show_s)
