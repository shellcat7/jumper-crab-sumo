"""The replay judgement (10/23, the user: the final's replay should end on the loser going out, as a
video review does): the moment of decision again, close and low, looking along the straw, in slow
motion, so the loser's foot is seen coming down beyond it. Look 5, as the bouts.

    blender -b -P verdict_shot.py -- DUMP OUT_DIR [SLOW] [BEFORE]

The loser and the foot are read off the dump: the first 勝負あり, the loser's foot tip lowest
beyond the straw at that frame. BEFORE dump frames before it (default 16) to two after are
rendered SLOW times slower (default 3; Blender's subframes between the dump's keys, which are
linear). Writes OUT_DIR/####.png and OUT_DIR/verdict.json (each PNG's foot position on the screen,
0..1 from the top left, and the PNG of the decision) for the circle drawn over it.
"""

import json
import math
import os
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sumo_scene as S  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
dump, out = argv[0], argv[1]
slow = int(argv[2]) if len(argv) > 2 else 3
before = int(argv[3]) if len(argv) > 3 else 16
os.makedirs(out, exist_ok=True)
RING = 0.8

bpy.ops.wm.read_factory_settings(use_empty=True)
D = dict(np.load(dump, allow_pickle=False))
names = [str(x) for x in D["dyn_name"]]
caps = [str(c) for c in D["caption"]]
dec = next(k for k, c in enumerate(caps) if c.startswith("勝負あり"))
loser = "robot" if "西の勝ち" in caps[dec] else "opponent"       # east is the robot
feet = [i for i, n in enumerate(names) if n.startswith(f"{loser}/") and n.endswith("foot_tip_link_visual")]
P = D["dyn_pos"][dec]
out_feet = [i for i in feet if math.hypot(*P[i, :2]) > RING]
foot = min(out_feet or feet, key=lambda i: P[i, 2])
print(f"[verdict] decision at {dec}: {caps[dec]}; {names[foot]} at r {math.hypot(*P[foot, :2]):.3f}")

k0, k1 = dec - before, min(len(caps) - 1, dec + 2)
frames = np.arange(k0, k1 + 1)
S.LOOK = int(os.environ.get("JUMPER_BLENDER_LOOK", "5"))
S.CROWD_CELL = 0.0
S.PORTRAITS = os.environ.get("JUMPER_BLENDER_PORTRAITS", "")
S.FRAMES = frames
M = S.mats()
S.build_arena(M, D)
S.import_robots(D, frames)
S._dressing(D, M)
fobj = next(ob for ob in bpy.data.objects if ob.name.startswith("geom_") and ob.name.endswith(names[foot]))

# the camera: low, just outside the straw and along it, looking back at the foot, so the straw
# runs across the picture and the foot comes down on the far side of it
f = Vector(P[foot].tolist())
u = Vector((f.x, f.y, 0)).normalized()
tang = Vector((-u.y, u.x, 0))
base = Vector(D["dyn_pos"][dec, names.index(f"{loser}/base_link_visual")].tolist())
side = 1 if tang.dot(base - f) < 0 else -1          # stand on the side away from the loser's body
cam = bpy.data.cameras.new("cam")
cam.lens = 50
cam.dof.use_dof = True
cam.dof.aperture_fstop = 3.5
co = bpy.data.objects.new("cam", cam)
bpy.context.scene.collection.objects.link(co)
sc = bpy.context.scene
sc.camera = co
co.location = f + tang * (0.55 * side) + u * 0.12 + Vector((0, 0, 0.13))
look = f - u * 0.06 + Vector((0, 0, 0.035))
co.rotation_mode = "QUATERNION"
co.rotation_quaternion = (look - co.location).to_track_quat("-Z", "Y")
cam.dof.focus_distance = (look - co.location).length

S.setup_render("eevee", int(os.environ.get("JUMPER_BLENDER_SAMPLES", "20")), "1920x1080", os.path.join(out, ""))
sc.view_settings.exposure = S.EXPOSURE_EEVEE["arena"]
marks, n, dec_png = [], 0, None
for j in range((k1 - k0) * slow + 1):
    t = 1 + j / slow                                 # Blender frame (1 = k0)
    sc.frame_set(int(t), subframe=t - int(t))
    n += 1
    sc.render.filepath = os.path.join(out, f"{n:04d}.png")
    if os.environ.get("VERDICT_STILL") and k0 + j / slow != dec:
        continue
    if not os.path.exists(sc.render.filepath):     # a crashed run goes on from the frames written
        bpy.ops.render.render(write_still=True)
    p = world_to_camera_view(sc, co, fobj.matrix_world.translation)
    marks.append([round(p.x, 4), round(1 - p.y, 4)])
    if k0 + j / slow == dec:
        dec_png = n
json.dump({"slow": slow, "first": int(k0), "decision": int(dec), "decision_png": dec_png, "foot": names[foot],
           "marks": marks}, open(os.path.join(out, "verdict.json"), "w"))
print(f"[verdict] {n} frames -> {out} (decision at {dec_png})")
