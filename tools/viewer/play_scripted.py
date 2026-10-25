"""Run the official scripts/play.py with commands from a schedule, and log the robot.

In replay the walking policy's commands come from the keyboard or a pad; with
nobody at the controls they are the training sampler's random draws. This wraps
the two command terms of jumper.posture-style tasks at run time so that every
step's command comes from a schedule file instead, and records the followed
robot (environment 0) at every control step. Falls (terminations other than
time-outs) are counted over every environment, so `--num_envs N` with
JUMPER_PLAY_DISTURB=1 measures how often a policy falls under pushes or wind.
Nothing in the upstream checkout is modified; every play.py argument still applies
(`--headless` for measuring).

Schedule (JSON): a list of segments, each held for `s` seconds; any key left out
keeps its neutral value. Velocities in the body frame, angles in radians,
height in metres above the ground (neutral 0.107):

    {"segments": [
      {"s": 2.0},
      {"s": 4.0, "vx": 0.3},
      {"s": 4.0, "vy": 0.3},
      {"s": 3.0, "wz": 0.5},
      {"s": 3.0, "twist": 0.3, "pitch": -0.1}
    ]}

    JUMPER_SCHEDULE   path to the schedule (required)
    JUMPER_LOG        CSV to write the per-step log to (optional)
    JUMPER_CAM_*      as in play_closeup.py

Run from the jumper checkout with its virtual environment:

    JUMPER_SCHEDULE=walk.json JUMPER_LOG=run.csv .venv/Scripts/python.exe \
        <this repo>/tools/viewer/play_scripted.py --task jumper.posture \
        --checkpoint <model.pt> --backend native --device cpu --no-viewer-ui
"""

from __future__ import annotations

import atexit
import csv
import json
import math
import os
import runpy
import sys
from pathlib import Path

NEUTRAL = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "twist": 0.0, "pitch": 0.0, "roll": 0.0, "height": 0.107}


def load_schedule(path: Path) -> list[tuple[float, dict]]:
    segs = json.loads(path.read_text(encoding="utf-8"))["segments"]
    out, t = [], 0.0
    for seg in segs:
        values = {**NEUTRAL, **{k: float(v) for k, v in seg.items() if k != "s"}}
        out.append((t, values))
        t += float(seg["s"])
    out.append((t, dict(NEUTRAL)))  # after the last segment: stand still
    return out


def play_script() -> Path:
    """The checkout's scripts/play.py, with its directory importable."""
    play = Path.cwd() / "scripts" / "play.py"
    if not play.is_file():
        raise SystemExit("run this from the jumper checkout: scripts/play.py not found")
    sys.path.insert(0, str(play.parent))
    return play


def run_play(play: Path) -> None:
    """Run play.py as if it had been started with this script's arguments."""
    sys.argv = [str(play)] + sys.argv[1:]
    runpy.run_path(str(play), run_name="__main__")


def install() -> None:
    """Hook the command terms, the step logger and the close camera; call before play.py."""
    sched_path = os.environ.get("JUMPER_SCHEDULE")
    if not sched_path:
        raise SystemExit("set JUMPER_SCHEDULE to a schedule JSON")
    schedule = load_schedule(Path(sched_path))
    total = schedule[-1][0]

    # Replay restores the step counter the checkpoint trained to (curricula count
    # in it), so the schedule's clock starts at the counter's first value seen.
    start: list[int] = []

    def elapsed(env) -> float:
        # Set by the first replay step (below), not by the command terms: they also
        # run while the environment is built, before the counter is restored.
        if not start:
            return 0.0
        return (int(env.common_step_counter) - start[0]) * env.step_dt

    def segment(t: float) -> int:
        """Index of the schedule segment in force at `t` (the last one is the stand after it)."""
        index = 0
        for i, (begin, _) in enumerate(schedule):
            if t >= begin:
                index = i
        return index

    def at(t: float) -> dict:
        return schedule[segment(t)][1]

    import torch
    from tasks.jumper.common.mdp import operator as op
    from tasks.jumper.posture.mdp import teleop as tp

    vel_compute = op.OperatorVelocityCommand.compute
    pose_compute = tp.TeleopPostureCommand.compute

    def scripted_vel(self, dt, env_ids=None):
        vel_compute(self, dt, env_ids)
        c = at(elapsed(self._env))
        self.vel_command_b[:, 0] = c["vx"]
        self.vel_command_b[:, 1] = c["vy"]
        self.vel_command_b[:, 2] = c["wz"]
        self.vel_command_w[:] = self.vel_command_b

    def scripted_pose(self, dt, env_ids=None):
        pose_compute(self, dt, env_ids)
        c = at(elapsed(self._env))
        self.posture_command[:] = torch.tensor(
            [c["twist"], c["pitch"], c["roll"], c["height"]], device=self.device)
        self.hold_to_band()

    op.OperatorVelocityCommand.compute = scripted_vel
    tp.TeleopPostureCommand.compute = scripted_pose

    rows: list[dict] = []
    log_path = os.environ.get("JUMPER_LOG")
    from mjlab.envs import ManagerBasedRlEnv

    step = ManagerBasedRlEnv.step
    #: Terminations other than time-outs (falls), summed over every environment,
    #: so a run with --num_envs N measures robustness rather than one robot's luck.
    falls = {"count": 0, "envs": 0}

    def logged_step(self, action):
        if not start:
            start.append(int(self.common_step_counter))
        out = step(self, action)
        falls["count"] += int(self.reset_terminated.sum())
        falls["envs"] = int(self.num_envs)
        t = elapsed(self)
        d = self.scene["robot"].data
        p = d.root_link_pos_w[0].tolist()
        w, x, y, z = d.root_link_quat_w[0].tolist()
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v = d.root_link_lin_vel_b[0].tolist()
        wz = d.root_link_ang_vel_b[0, 2].item()
        c = at(t)
        rows.append({"t": round(t, 4), "x": p[0], "y": p[1], "z": p[2], "roll": roll,
                     "pitch": pitch, "yaw": yaw, "vx": v[0], "vy": v[1], "wz": wz,
                     "fell": int(self.reset_terminated[0]),
                     # Every environment's horizontal speed, averaged: while the
                     # schedule says stand still, this is how hard wind shoves them.
                     "speed_all": float(d.root_link_lin_vel_b[:, :2].norm(dim=1).mean()),
                     "vx_all": float(d.root_link_lin_vel_b[:, 0].mean()),
                     "vy_all": float(d.root_link_lin_vel_b[:, 1].mean()),
                     "wz_all": float(d.root_link_ang_vel_b[:, 2].mean()),
                     # 10/15, the body's quality, every environment averaged: the bob
                     # (vertical speed and roll/pitch rates, squared) and how far the
                     # achieved planar velocity is from the commanded one (squared).
                     "vz2_all": float(d.root_link_lin_vel_b[:, 2].square().mean()),
                     "wxy2_all": float(d.root_link_ang_vel_b[:, :2].square().sum(dim=1).mean()),
                     "err2_all": float(((d.root_link_lin_vel_b[:, 0] - c["vx"]).square()
                                        + (d.root_link_lin_vel_b[:, 1] - c["vy"]).square()).mean()),
                     "werr2_all": float((d.root_link_ang_vel_b[:, 2] - c["wz"]).square().mean()),
                     "seg": segment(t),
                     **{f"cmd_{k}": c[k] for k in ("vx", "vy", "wz", "twist", "pitch", "roll", "height")}})
        return out

    ManagerBasedRlEnv.step = logged_step

    def write_log():
        if rows:
            print(f"[scripted] falls: {falls['count']} across {falls['envs']} environments "
                  f"in {rows[-1]['t']:.1f} s")
            still = [r["speed_all"] for r in rows
                     if r["cmd_vx"] == r["cmd_vy"] == r["cmd_wz"] == 0.0 and r["t"] >= 1.0]
            if still:
                print(f"[scripted] mean speed while told to stand (after 1 s): "
                      f"{sum(still) / len(still):.4f} m/s over {len(still)} steps")
            late = [r for r in rows if r["t"] >= 1.0]
            if late:
                n = len(late)
                print(f"[scripted] body: bob {1000 * math.sqrt(sum(r['vz2_all'] for r in late) / n):.1f} mm/s, "
                      f"roll/pitch rate {math.degrees(math.sqrt(sum(r['wxy2_all'] for r in late) / n)):.1f} deg/s, "
                      f"velocity error {math.sqrt(sum(r['err2_all'] for r in late) / n):.3f} m/s, "
                      f"turn error {math.sqrt(sum(r['werr2_all'] for r in late) / n):.3f} rad/s (rms, after 1 s)")
            # Commanded against achieved, every environment averaged, over the second
            # half of each segment (the first half is the robot getting up to speed).
            for seg in sorted({r["seg"] for r in rows}):
                part = [r for r in rows if r["seg"] == seg]
                part = part[len(part) // 2:]
                if not part:
                    continue
                mean = {k: sum(r[k] for r in part) / len(part) for k in ("vx_all", "vy_all", "wz_all")}
                c = part[0]
                print(f"[scripted] segment {seg}: cmd vx {c['cmd_vx']:+.2f} vy {c['cmd_vy']:+.2f} "
                      f"wz {c['cmd_wz']:+.2f} -> achieved vx {mean['vx_all']:+.3f} "
                      f"vy {mean['vy_all']:+.3f} wz {mean['wz_all']:+.3f}")
        if log_path and rows:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
                w.writeheader()
                w.writerows(rows)
            tilt = max(math.degrees(max(abs(r["roll"]), abs(r["pitch"]))) for r in rows)
            print(f"[scripted] {len(rows)} steps logged to {log_path}; max tilt {tilt:.1f} deg")

    atexit.register(write_log)

    from mjrl.viewer import live

    distance = os.environ.get("JUMPER_CAM_DISTANCE")
    if distance:
        aim = live.LiveViewer._aim_camera

        def aim_closer(self, model) -> None:
            aim(self, model)
            cam = self._handle.cam
            cam.distance = float(distance)
            cam.elevation = float(os.environ.get("JUMPER_CAM_ELEVATION", "-20"))
            if os.environ.get("JUMPER_CAM_AZIMUTH"):
                cam.azimuth = float(os.environ["JUMPER_CAM_AZIMUTH"])

        live.LiveViewer._aim_camera = aim_closer

    print(f"[scripted] {len(schedule) - 1} segments, {total:.1f} s, from {sched_path}")


def main() -> None:
    play = play_script()
    install()
    run_play(play)


if __name__ == "__main__":
    main()
