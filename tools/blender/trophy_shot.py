"""The champion with the cup (10/23, the user: after the decision, the winner holding the cup, not
just a caption): the winner of a dump alone on the ring in the claws' V of the ceremony, an
ORIGINAL gold cup (no replica of any real trophy) raised between the claws, the live crowd behind,
the camera moving slowly in from the front and a little below; the champion still (10/23: no
hopping -- sumo's courtesy). Look 5, as the bouts.

    blender -b -P trophy_shot.py -- DUMP OUT_DIR ENTITY [FRAME] [N_FRAMES]

ENTITY: robot (east) or opponent (west); FRAME: the dump frame of the pose (default: the middle of
the ceremony's V); N_FRAMES: the shot's length (default 105 = 3.5 s). Writes OUT_DIR/####.png.
"""

import math
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sumo_scene as S  # noqa: E402

argv = sys.argv[sys.argv.index("--") + 1:]
dump, out, ent = argv[0], argv[1], argv[2]
k = int(argv[3]) if len(argv) > 3 else 95
N = int(argv[4]) if len(argv) > 4 else 105
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
D = dict(np.load(dump, allow_pickle=False))
names = [str(x) for x in D["dyn_name"]]
S.LOOK = int(os.environ.get("JUMPER_BLENDER_LOOK", "5"))
S.CROWD_CELL = 0.0
S.PORTRAITS = os.environ.get("JUMPER_BLENDER_PORTRAITS", "")
S.FRAMES = np.full(N, k)                       # the crowd held at that moment
M = S.mats()
S.build_arena(M, D)
S.import_robots(D, np.array([k]))
S._dressing(D, M)
# the winner alone: the other wrestler and the gyoji go
for ob in list(bpy.data.objects):
    nm = ob.name
    if nm.startswith("geom_") and f"_{ent}/" not in nm:
        bpy.data.objects.remove(ob)
others = tuple(f"{e}_" for e in ("robot", "opponent", "gyoji") if e != ent)
for ob in list(bpy.data.objects):            # the others' topknot / eboshi (_dressing: "<entity>_<part>")
    if ob.name.startswith(others):
        bpy.data.objects.remove(ob)

gi = names.index(f"{ent}/upper_shell_link_visual")
c = Vector(D["dyn_pos"][k, gi].tolist())
yaw = math.atan2(*reversed(Vector(D["dyn_mat"][k, gi].reshape(3, 3)[:, 0].tolist()).xy[:]))
front = yaw + math.radians(30)                  # where the display faces (portrait_renders, turned 180)
fwd = Vector((math.cos(front), math.sin(front), 0.0))

# the cup: a lathed gold body on a stem and foot, two handles; about 13 cm tall, raised between
# the claws above the shell
gold = S._principled("cup_gold", (0.95, 0.72, 0.28), rough=0.22, metal=1.0, coat=0.4)
prof = [(0.000, 0.000), (0.034, 0.000), (0.034, 0.006), (0.012, 0.014), (0.009, 0.040), (0.016, 0.048),
        (0.036, 0.062), (0.046, 0.085), (0.050, 0.112), (0.052, 0.128), (0.048, 0.130), (0.044, 0.112),
        (0.040, 0.088)]
cup = S._lathe("cup", prof, gold, segs=64)
base = c + Vector((0, 0, 0.030)) + fwd * 0.045   # resting on the shell, in front of the topknot
cup.location = base
for side in (-1, 1):
    bpy.ops.mesh.primitive_torus_add(major_radius=0.022, minor_radius=0.0045,
                                     location=base + Vector((0, 0, 0.095)))
    h = bpy.context.object
    h.name = f"cup_handle_{side}"
    h.data.materials.append(gold)
    h.rotation_euler = (math.pi / 2, 0, front + math.pi / 2)
    side_dir = Vector((-math.sin(front), math.cos(front), 0.0)) * side
    h.location = base + Vector((0, 0, 0.095)) + side_dir * 0.052
# no hop, no bow (10/23 21:50, the user: hopping for joy is sweet, but sumo honours courtesy; a bow
# tilted about the front feet lifted the back legs off the clay and read as a hop again): the
# champion stands still with the cup, and only the camera moves
# a warm spot from above the shomen side on the champion
L = bpy.data.lights.new("champion_spot", "SPOT")
L.energy, L.spot_size, L.spot_blend, L.color = 220, math.radians(28), 0.5, (1.0, 0.9, 0.75)
lo = bpy.data.objects.new("champion_spot", L)
bpy.context.scene.collection.objects.link(lo)
lo.location = c + fwd * 0.9 + Vector((0, 0, 1.6))
lo.rotation_mode = "QUATERNION"
lo.rotation_quaternion = (c + Vector((0, 0, 0.08)) - lo.location).to_track_quat("-Z", "Y")

# the camera: from the front, a little below the cup, moving in
cam = bpy.data.cameras.new("cam")
cam.lens = 50
cam.dof.use_dof = True
cam.dof.aperture_fstop = 2.8
co = bpy.data.objects.new("cam", cam)
bpy.context.scene.collection.objects.link(co)
sc = bpy.context.scene
sc.camera = co
look = c + Vector((0, 0, 0.04))
for i in range(N):
    s = i / max(1, N - 1)
    s = s * s * (3 - 2 * s)
    dist, h, az = 1.75 - 0.55 * s, 0.17 + 0.03 * s, front + math.radians(12 - 14 * s)
    co.location = c + Vector((math.cos(az) * dist, math.sin(az) * dist, h))
    co.rotation_mode = "QUATERNION"
    co.rotation_quaternion = (look - co.location).to_track_quat("-Z", "Y")
    cam.dof.focus_distance = (look - co.location).length
    co.keyframe_insert("location", frame=i + 1)
    co.keyframe_insert("rotation_quaternion", frame=i + 1)
    cam.keyframe_insert("dof.focus_distance", frame=i + 1)
sc.frame_start, sc.frame_end = 1, N
S.setup_render("eevee", int(os.environ.get("JUMPER_BLENDER_SAMPLES", "20")), "1920x1080", os.path.join(out, ""))
sc.view_settings.exposure = S.EXPOSURE_EEVEE["arena"]
if os.environ.get("TROPHY_STILL"):
    sc.frame_set(int(os.environ["TROPHY_STILL"]))
    sc.render.filepath = os.path.join(out, "still.png")
    bpy.ops.render.render(write_still=True)
else:
    sc.render.use_overwrite = False          # a crashed run goes on from the frames written
    sc.render.use_placeholder = True
    bpy.ops.render.render(animation=True)
print(f"[trophy] {ent} at frame {k}: {N} frames -> {out}")
