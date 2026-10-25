"""Record a replay of the official scripts/play.py to video, at real-time speed.

A replay with the window open runs slower than real time when the PC is busy (10/9:
about 0.05x for jumper.posture), so a screen recording shows slow motion. This draws
frames off screen instead: play.py runs headless and unpaced, and every 1/FPS
seconds of *simulated* time environment 0 is drawn with mujoco.Renderer from a
camera tracking its base and appended to an mp4. The video plays at real speed
however fast or slow the replay ran. Nothing in the upstream checkout is modified.

With JUMPER_SCHEDULE set, play_scripted.py's hooks are installed first, so the
robot follows the schedule (and JUMPER_LOG still writes the per-step log).

    JUMPER_VIDEO      output .mp4 (required)
    JUMPER_FPS        frames per simulated second (default 30)
    JUMPER_SIZE       WIDTHxHEIGHT (default 1280x720)
    JUMPER_CAM_DISTANCE, JUMPER_CAM_ELEVATION, JUMPER_CAM_AZIMUTH
                      tracking camera: metres from the base, degrees (default 0.9, -20, 135)
    JUMPER_CAM_LOOKAT x,y,z: a fixed free camera looking at this point instead of
                      tracking the base (e.g. 0,0,0.05 for the sumo ring)
    JUMPER_CAM_FOLLOW=bout  with JUMPER_CAM_LOOKAT: the free camera's point drifts
                      after the bout -- 60% of the way from the ring's centre to
                      the two wrestlers' midpoint, smoothed over 0.6 s -- so a bout
                      pushed to the far straw stays in frame (jumper.sumo)
    JUMPER_CAM_INTRO "seconds,distance,elevation,lookat_z": with JUMPER_CAM_LOOKAT,
                      open on a wide view of the hall and close in on the free
                      camera over this many seconds (jumper.sumo's audience),
                      before the world starts: the wrestlers wait on their marks
                      while the camera closes in, and the bout begins after it
                      (the user, 10/11: the bout had begun mid-zoom)
    JUMPER_SUMO_NAMES "east,west": ring names for a strip along the bottom with the
                      score of the bouts so far in this video (jumper.sumo with a
                      gyoji or the referee)
    JUMPER_BLENDER_DUMP  also write every frame's geom poses to this .npz for a Blender
                      render (tools/viewer/blender_dump.py, tools/blender/sumo_scene.py)
    JUMPER_SCHEDULE, JUMPER_LOG   as in play_scripted.py
    JUMPER_SUMO_DRIVE, JUMPER_SUMO_LOG   as in play_sumo.py (installs its hooks)

Native backend only: it reads the followed environment's host MjData. Run from the
jumper checkout with its virtual environment, with --headless and --steps:

    JUMPER_VIDEO=walk.mp4 JUMPER_SCHEDULE=walk_tour.json .venv/Scripts/python.exe \
        <this repo>/tools/viewer/render_offscreen.py --task jumper.posture \
        --checkpoint <model.pt> --backend native --device cpu --headless --steps 5400
"""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import play_scripted  # noqa: E402


_FONT = "C:/Windows/Fonts/BIZ-UDMinchoM.ttc"


def _font(size):
    from PIL import ImageFont

    try:
        return ImageFont.truetype(_FONT, size)
    except OSError:
        return ImageFont.load_default()


def _caption_state(referee):
    """The call shown now (the latest within 1.6 s, or the decision) and the score."""
    now = float(referee.time[0])
    starts = [i for i, (_, text) in enumerate(referee.calls) if text == "見合って"]
    bout_calls = referee.calls[starts[-1]:] if starts else referee.calls
    current = [(t, text) for t, text in bout_calls if t <= now + 1e-6]
    text = ""
    if current:
        t, last = current[-1]
        if now - t <= 1.6 or last.startswith(("勝負あり", "同体")):
            text = last
    wins = [sum(1 for _, c in referee.calls if c.startswith(f"勝負あり　{side}")) for side in ("東", "西")]
    return text, wins


def _caption(frame, referee):
    """The gyoji's latest call over the top of the frame (environment 0), and with
    JUMPER_SUMO_NAMES the two ring names and the score along the bottom.

    A call shows for 1.6 s; the decision stays until the next bout.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")
    now = float(referee.time[0])
    # Calls are kept per bout from 0 s; the list runs across bouts, so the current
    # bout's calls are the ones after its last "見合って".
    starts = [i for i, (_, text) in enumerate(referee.calls) if text == "見合って"]
    bout_calls = referee.calls[starts[-1]:] if starts else referee.calls
    current = [(t, text) for t, text in bout_calls if t <= now + 1e-6]
    if current:
        t, text = current[-1]
        if now - t <= 1.6 or text.startswith(("勝負あり", "同体")):
            font = _font(54)
            w = draw.textlength(text, font=font)
            x = (img.width - w) / 2
            draw.rectangle([x - 24, 22, x + w + 24, 98], fill=(20, 12, 30, 170))
            draw.text((x, 30), text, font=font, fill=(250, 240, 210, 255))
    names = os.environ.get("JUMPER_SUMO_NAMES")
    if names and "," in names:
        east, west = names.split(",", 1)
        wins = [sum(1 for _, c in referee.calls if c.startswith(f"勝負あり　{side}"))
                for side in ("東", "西")]
        font = _font(40)
        y0, y1 = img.height - 84, img.height - 24
        draw.rectangle([0, y0, img.width, y1], fill=(20, 12, 30, 150))
        left = f"東　{east}"
        right = f"{west}　西"
        score = f"{wins[0]} − {wins[1]}"
        draw.rectangle([24, y0 + 14, 44, y1 - 14], fill=(34, 69, 184, 255))      # east mawashi
        draw.text((56, y0 + 8), left, font=font, fill=(250, 240, 210, 255))
        wr = draw.textlength(right, font=font)
        draw.rectangle([img.width - 44, y0 + 14, img.width - 24, y1 - 14], fill=(26, 133, 77, 255))
        draw.text((img.width - 56 - wr, y0 + 8), right, font=font, fill=(250, 240, 210, 255))
        ws = draw.textlength(score, font=font)
        draw.text(((img.width - ws) / 2, y0 + 8), score, font=font, fill=(255, 255, 255, 255))
    return np.asarray(img)


def install_recorder() -> None:
    import imageio_ffmpeg
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv

    out = os.environ.get("JUMPER_VIDEO")
    # JUMPER_NO_FRAMES=1 (10/21 night): the replay without any picture -- no OpenGL, no video, the
    # physics, the gyoji's calls and JUMPER_SUMO_LOG as ever (a fast search for a bout worth filming;
    # the frames never touch the simulation, so a seed plays the same with them)
    no_frames = os.environ.get("JUMPER_NO_FRAMES") == "1"
    if not out and not no_frames:
        raise SystemExit("set JUMPER_VIDEO to the .mp4 to write")
    fps = float(os.environ.get("JUMPER_FPS", "30"))
    width, height = (int(v) for v in os.environ.get("JUMPER_SIZE", "1280x720").split("x"))
    distance = float(os.environ.get("JUMPER_CAM_DISTANCE", "0.9"))
    elevation = float(os.environ.get("JUMPER_CAM_ELEVATION", "-20"))
    azimuth = float(os.environ.get("JUMPER_CAM_AZIMUTH", "135"))

    state: dict = {}

    def open_renderer(env) -> None:
        sim = env.sim
        if not hasattr(sim, "env_mjdata"):
            raise SystemExit("render_offscreen.py needs --backend native (it reads host MjData)")
        model = sim.render_model
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        cam = mujoco.MjvCamera()
        lookat = os.environ.get("JUMPER_CAM_LOOKAT")
        if lookat:
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [float(v) for v in lookat.split(",")]
        else:
            free = int(mujoco.mjtJoint.mjJNT_FREE)
            base = next(i for i in range(model.nbody)
                        if model.body_jntnum[i] > 0 and model.jnt_type[model.body_jntadr[i]] == free)
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = base
        cam.distance, cam.elevation, cam.azimuth = distance, elevation, azimuth
        writer = None
        if not no_frames:
            writer = imageio_ffmpeg.write_frames(
                out, (width, height), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                output_params=["-crf", "18"], macro_block_size=8)
            writer.send(None)
        opt = mujoco.MjvOption()
        # Geom groups 0-2 as MuJoCo shows them by default, plus 3: decorations a task
        # keeps out of its sensors' sight (jumper.sumo's dohyo).
        for g in (0, 1, 2, 3):
            opt.geomgroup[g] = 1
        state["opt"] = opt
        if os.environ.get("JUMPER_CAM_FOLLOW") == "bout" and lookat:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{e}/base_link")
                   for e in ("robot", "opponent")]
            if min(ids) >= 0:
                state["follow"] = (ids, cam.lookat.copy())
        intro = os.environ.get("JUMPER_CAM_INTRO")
        if intro and lookat:
            secs, d0, e0, z0 = (float(v) for v in intro.split(","))
            state["intro"] = (secs, d0, e0, z0, distance, elevation, float(cam.lookat[2]))
        dump = os.environ.get("JUMPER_BLENDER_DUMP")
        if dump:
            # tools/viewer/blender_dump.py: the poses for a Blender render of this replay.
            from blender_dump import Dumper

            state["dump"] = Dumper(dump)
        state.update(sim=sim, model=model, data=mujoco.MjData(model), cam=cam,
                     # A hall of spectators (jumper.sumo's audience) is tens of
                     # thousands of geoms; the default scene holds 10 000.
                     renderer=None if no_frames else mujoco.Renderer(model, height, width,
                                                                     max_geom=max(10000, model.ngeom + 2000)),
                     writer=writer,
                     steps=0, frames=0)

    def draw(env, intro_k: float | None = None) -> None:
        _, live = state["sim"].env_mjdata(0)
        data, model = state["data"], state["model"]
        data.qpos[:], data.qvel[:] = live.qpos, live.qvel
        data.mocap_pos[:], data.mocap_quat[:] = live.mocap_pos, live.mocap_quat
        mujoco.mj_forward(model, data)
        decorate = getattr(env, "sumo_decorate", None)   # jumper.sumo's gunbai
        if decorate is not None:
            decorate(model, data)
        if "follow" in state:
            ids, home = state["follow"]
            mid = (data.xpos[ids[0]] + data.xpos[ids[1]]) / 2
            target = home.copy()
            target[:2] = home[:2] + 0.6 * (mid[:2] - home[:2])
            cam = state["cam"]
            cam.lookat[:] = cam.lookat + min(1.0, 1.0 / (fps * 0.6)) * (target - cam.lookat)
        if "intro" in state:
            secs, d0, e0, z0, d1, e1, z1 = state["intro"]
            k = 1.0 if intro_k is None else intro_k
            k = k * k * (3 - 2 * k)
            cam = state["cam"]
            cam.distance, cam.elevation = d0 + (d1 - d0) * k, e0 + (e1 - e0) * k
            cam.lookat[2] = z0 + (z1 - z0) * k
        referee = getattr(env, "sumo_referee", None)     # jumper.sumo's gyoji calls
        if "dump" in state:
            text, wins = _caption_state(referee) if referee is not None and referee.calls else ("", (0, 0))
            state["dump"].frame(model, data, state["opt"], state["cam"], state["frames"] / fps, text, wins)
        r = state["renderer"]
        if r is None:
            state["frames"] += 1
            return
        r.update_scene(data, camera=state["cam"], scene_option=state["opt"])
        frame = r.render()
        if referee is not None and referee.calls:
            frame = _caption(frame, referee)
        state["writer"].send(frame)
        state["frames"] += 1

    step = ManagerBasedRlEnv.step

    def recorded_step(self, action):
        if not state:
            open_renderer(self)
            if "intro" in state:
                # The opening move, the world held still: frames of the starting
                # state while the camera closes in. A hall's crowd keeps its own
                # clock meanwhile (`sumo_clock`, then `sumo_clock_offset`).
                n = max(1, int(round(state["intro"][0] * fps)))
                for i in range(n):
                    self.sumo_clock = i / fps
                    draw(self, intro_k=i / n)
                self.sumo_clock = None
                self.sumo_clock_offset = n / fps
            state["offset"] = state["frames"]
            draw(self)
        result = step(self, action)
        state["steps"] += 1
        # Draw whenever simulated time passes the next frame's time.
        if state["steps"] * self.step_dt >= (state["frames"] - state["offset"]) / fps:
            draw(self)
        return result

    ManagerBasedRlEnv.step = recorded_step

    def close() -> None:
        if state:
            if state["writer"] is not None:
                state["writer"].close()
                state["renderer"].close()
            if "dump" in state:
                state["dump"].write()
            print(f"[offscreen] {state['frames']} frames ({state['frames'] / fps:.1f} s at "
                  f"{fps:g} fps, {width}x{height}) written to {out}")

    atexit.register(close)


def main() -> None:
    play = play_scripted.play_script()
    if os.environ.get("JUMPER_SCHEDULE"):
        play_scripted.install()
    if os.environ.get("JUMPER_SUMO_DRIVE") or os.environ.get("JUMPER_SUMO_LOG"):
        import play_sumo

        play_sumo.install()
    install_recorder()
    play_scripted.run_play(play)


if __name__ == "__main__":
    main()
