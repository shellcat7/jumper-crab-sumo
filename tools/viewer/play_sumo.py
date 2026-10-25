"""Run the official scripts/play.py on jumper.sumo and measure the bout.

Hooks, installed at run time (nothing in the upstream checkout is modified):

- JUMPER_SUMO_DRIVE=SPEED: the learner's velocity command is replaced every step
  by "turn to face the opponent and walk at it at SPEED m/s" -- the same steering
  the opponent uses -- so two copies of a walking policy can be put into the ring
  against each other without any training. Unset, the learner follows play.py's
  own command (keyboard / pad, or the sampler's random draws).
- Every control step, environment 0 is logged: both bases (x, y, yaw), their
  distance, and the number of contacts between a geom of one robot and a geom of
  the other (native backend: read from the host MjData). Bouts that end are
  counted by reason (which termination fired), over every environment.

    JUMPER_SUMO_LOG   CSV for the per-step log (optional)

Run from the jumper checkout with its virtual environment, e.g.

    JUMPER_SUMO_DRIVE=0.3 .venv/Scripts/python.exe <repo>/tools/viewer/play_sumo.py \
        --task jumper.sumo --checkpoint <model.pt> --backend native --device cpu --headless --steps 2000
"""

from __future__ import annotations

import atexit
import csv
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_scripted  # noqa: E402


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def install() -> None:
    import torch
    from mjlab.envs import ManagerBasedRlEnv

    seed = os.environ.get("JUMPER_SEED")
    if seed:
        # A fixed environment seed (resets, randomization): the same run again, so a
        # headless search for a bout can be followed by a render of that very bout.
        init = ManagerBasedRlEnv.__init__

        def seeded(self, cfg, *args, **kwargs):
            cfg.seed = int(seed)
            init(self, cfg, *args, **kwargs)

        ManagerBasedRlEnv.__init__ = seeded

    drive = os.environ.get("JUMPER_SUMO_DRIVE")
    if drive:
        from tasks.jumper.common.mdp import operator as op
        from tasks.jumper.posture.mdp import teleop as tp
        from tasks.jumper.sumo.mdp import SHIKIRI_POSTURE, bout, hold_mark, shikiri_or
        from tasks.jumper.sumo.opponent import approach_command

        speed = float(drive)
        compute = op.OperatorVelocityCommand.compute
        pose_compute = tp.TeleopPostureCommand.compute

        def driven(self, dt, env_ids=None):
            compute(self, dt, env_ids)
            env = self._env
            if "opponent" in env.scene.entities:
                state = bout(env)
                state.update()
                cmd = approach_command(env.scene["robot"], env.scene["opponent"], speed)
                cmd = shikiri_or(cmd, state.started, hold_mark(env, "robot"))
                self.vel_command_b[:] = cmd
                self.vel_command_w[:] = cmd

        def posed(self, dt, env_ids=None):
            # Shikiri: crouched, claws down, until the tachiai (sumo/mdp.py BoutState).
            pose_compute(self, dt, env_ids)
            env = self._env
            if "opponent" in env.scene.entities:
                state = bout(env)
                state.update()
                neutral = torch.tensor([0.0, 0.0, 0.0, 0.10647], device=self.device)
                crouch = neutral + torch.tensor(SHIKIRI_POSTURE, device=self.device)
                self.posture_command[:] = shikiri_or(
                    neutral.expand(self.num_envs, 4), state.started, crouch)
                self.hold_to_band()

        op.OperatorVelocityCommand.compute = driven
        tp.TeleopPostureCommand.compute = posed

    rows: list[dict] = []
    ends: Counter = Counter()
    state: dict = {}
    log_path = os.environ.get("JUMPER_SUMO_LOG")
    step = ManagerBasedRlEnv.step

    def robot_pairs(env):
        """Body-id sets of the two robots in env 0's model (native backend)."""
        sim = env.sim
        if not hasattr(sim, "env_mjdata"):
            return None
        model, _ = sim.env_mjdata(0)
        import mujoco

        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
        mine = {i for i, n in enumerate(names) if n.startswith("robot/")}
        theirs = {i for i, n in enumerate(names) if n.startswith("opponent/")}
        gyoji = {i for i, n in enumerate(names) if n.startswith("gyoji/")}
        return mine, theirs, gyoji

    def logged(self, action):
        out = step(self, action)
        if "pairs" not in state:
            state["env"] = self
            state["pairs"] = robot_pairs(self)
            state["t0"] = int(self.common_step_counter)
        tm = self.termination_manager
        for name in tm.active_terms:
            fired = tm.get_term(name)
            if fired is not None and bool(fired.any()):
                ends[name] += int(fired.sum())
        a = self.scene["robot"].data
        b = self.scene["opponent"].data
        pa = a.root_link_pos_w[0].tolist()
        pb = b.root_link_pos_w[0].tolist()
        contacts = bumps = ""
        if state["pairs"] is not None:
            import mujoco

            model, data = self.sim.env_mjdata(0)
            mine, theirs, gyoji = state["pairs"]
            wrestlers = mine | theirs
            n = g = 0
            for c in data.contact[: data.ncon]:
                b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
                if (b1 in mine and b2 in theirs) or (b1 in theirs and b2 in mine):
                    n += 1
                if (b1 in gyoji and b2 in wrestlers) or (b1 in wrestlers and b2 in gyoji):
                    g += 1
            contacts = n
            bumps = g if gyoji else ""
            # Which of the robot's bodies touch the ground (JUMPER_SUMO_GROUND=1).
            if os.environ.get("JUMPER_SUMO_GROUND") == "1":
                ground = state.setdefault("ground", Counter())
                if "terrain_body" not in state:
                    state["terrain_body"] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
                started = bool(getattr(self, "sumo_bout", None) is not None and self.sumo_bout.started[0])
                for c in data.contact[: data.ncon]:
                    for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                        if model.geom_bodyid[g2] in (0, state["terrain_body"]) and model.geom_bodyid[g1] in mine:
                            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1]))
                            ground[("bout" if started else "shikiri", name)] += 1
        rows.append({
            "t": round((int(self.common_step_counter) - state["t0"]) * self.step_dt, 4),
            "robot_x": pa[0], "robot_y": pa[1], "robot_z": pa[2],
            "robot_yaw": _yaw(a.root_link_quat_w[0].tolist()),
            "opp_x": pb[0], "opp_y": pb[1], "opp_z": pb[2],
            "opp_yaw": _yaw(b.root_link_quat_w[0].tolist()),
            "dist": math.hypot(pa[0] - pb[0], pa[1] - pb[1]),
            "contacts": contacts,
            "reset": int(self.reset_buf[0]),
            "started": int(getattr(self, "sumo_bout", None) is not None and bool(self.sumo_bout.started[0])),
        })
        # Claws (the "hands" of the shikiri): how far forward of the shikiri line each
        # wrestler's finger tips and palm pads reach, along the ring's x axis (the
        # line is at x = -/+ line_x for east / west, `env_cfg._LINE_GAP`).
        if "claws" not in state:
            ids = {}
            for name, ent in (("robot", a), ("opp", b)):
                names = list(self.scene["robot" if name == "robot" else "opponent"].body_names)
                ids[name] = {
                    "tip": [names.index(n) for n in ("LF_finger_tip_link", "RF_finger_tip_link") if n in names],
                    "pad": [names.index(n) for n in ("LF_palm_pad_b_link", "RF_palm_pad_b_link") if n in names],
                }
            state["claws"] = ids
        for name, ent, sign in (("robot", a, 1.0), ("opp", b, -1.0)):
            pos = ent.body_link_pos_w[0]
            for part, idx in state["claws"][name].items():
                if idx:
                    # Furthest towards the centre: max x for east, min x for west.
                    rows[-1][f"{name}_{part}_x"] = max(sign * float(pos[i, 0]) for i in idx) * sign
                    # ... and the lower of the two above the ground.
                    rows[-1][f"{name}_{part}_z"] = min(float(pos[i, 2]) for i in idx)
        if getattr(self, "sumo_bout", None) is not None:
            # Shikiri: time both pairs of claws have been down, and time since the reset.
            rows[-1].update({"hands": round(float(self.sumo_bout.hands[0]), 4),
                             "shikiri": round(float(self.sumo_bout.elapsed[0]), 4)})
        if "gyoji" in self.scene.entities:
            # The physical gyoji (JUMPER_SUMO_GYOJI=1): where it is, and wrestler contacts.
            g = self.scene["gyoji"].data
            pg = g.root_link_pos_w[0].tolist()
            rows[-1].update({
                "gyoji_x": pg[0], "gyoji_y": pg[1], "gyoji_z": pg[2],
                "gyoji_yaw": _yaw(g.root_link_quat_w[0].tolist()),
                "gyoji_clear": min(math.hypot(pg[0] - pa[0], pg[1] - pa[1]),
                                   math.hypot(pg[0] - pb[0], pg[1] - pb[1])),
                "gyoji_contacts": bumps,
            })
        return out

    ManagerBasedRlEnv.step = logged

    def report():
        if not rows:
            return
        touching = [r for r in rows if isinstance(r["contacts"], int) and r["contacts"] > 0]
        print(f"[sumo] {len(rows)} steps; min distance {min(r['dist'] for r in rows):.3f} m; "
              f"steps with robot-robot contact {len(touching)} (max {max((r['contacts'] for r in touching), default=0)} contacts)")
        print(f"[sumo] bouts ended (all environments), by termination: {dict(ends) or 'none'}")
        if state.get("ground"):
            print("[sumo] robot bodies on the ground (steps): " + ", ".join(
                f"{phase}:{name.split('/')[-1]} {n}" for (phase, name), n in sorted(state["ground"].items())))
        referee = getattr(state.get("env"), "sumo_referee", None)
        if referee is not None:
            gy = [r for r in rows if "gyoji_clear" in r]
            if gy:
                bumped = sum(1 for r in gy if isinstance(r["gyoji_contacts"], int) and r["gyoji_contacts"] > 0)
                print(f"[sumo] gyoji: nearest wrestler min {min(r['gyoji_clear'] for r in gy):.3f} m, "
                      f"mean {sum(r['gyoji_clear'] for r in gy) / len(gy):.3f} m; steps touching a wrestler {bumped}; "
                      f"lowest base {min(r['gyoji_z'] for r in gy):.3f} m")
            # With JUMPER_SUMO_GYOJI=1: environment 0's calls, and the decisions so far.
            print("[sumo] gyoji (env 0): " + " / ".join(f"{t:.2f}s {text}" for t, text in referee.calls))
            from tasks.jumper.sumo.referee import tally_text

            # E = east (the learner) won, W = west (the opponent), D = dotai.
            print(f"[sumo] kimarite (all environments): {tally_text(referee) or 'none'}")
        if log_path:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            print(f"[sumo] log written to {log_path}")

    atexit.register(report)


def main() -> None:
    play = play_scripted.play_script()
    install()
    play_scripted.run_play(play)


if __name__ == "__main__":
    main()
