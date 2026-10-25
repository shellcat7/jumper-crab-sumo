"""Champions' portraits for the hall's walls (10/20): our own wrestlers -- the Jumper robots
of a dump -- rendered one at a time in a studio pose (three-quarter view, soft key light, a
plain warm backdrop), as a sumo hall's portraits of its champions are. Made-up wrestlers
only: no real person, name or venue. tools/blender/portraits.py then frames them.

    blender -b -P portrait_renders.py -- DUMP OUT_DIR [frames...]

Writes OUT_DIR/portrait_<entity>_<frame>.png (robot = east, opponent = west) for each frame.
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
dump, out = argv[0], argv[1]
D = dict(np.load(dump, allow_pickle=False))
n = len(D["t"])
frames = [int(x) for x in argv[2:]] or [int(n * 0.1), int(n * 0.3)]
os.makedirs(out, exist_ok=True)
names = [str(x) for x in D["dyn_name"]]

for k in frames:
    for ent in ("robot", "opponent"):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        S._MATS.clear()
        S.LOOK = 5
        M = S.mats()
        S.import_robots(D, np.array([k]))
        S._dressing(D, M)
        sc = bpy.context.scene
        sc.frame_set(1)
        # only this wrestler
        for ob in list(bpy.data.objects):
            if ob.name.startswith("geom_") and f"_{ent}/" not in ob.name:
                bpy.data.objects.remove(ob)
        gi = names.index(f"{ent}/upper_shell_link_visual")
        c = Vector(D["dyn_pos"][k, gi].tolist())
        # a backdrop: a curved cyclorama in warm grey, a floor of clay
        # 10/23 (the user: the legs sank into the floor): the floor's top at the feet -- the
        # lowest of this wrestler's geoms (foot tips, ~3 mm above the sim's ground) -- not 12 cm
        # under the shell, which was 2.6 cm too high
        ids = [i for i, n in enumerate(names) if n.startswith(f"{ent}/")]
        ground = float(D["dyn_pos"][k, ids, 2].min()) - 0.004
        S.box("floor", (c.x, c.y, ground - 0.01), (2, 2, 0.01), M["clay"])   # half-extents: top at ground
        bpy.ops.mesh.primitive_cylinder_add(radius=1.4, depth=2.0, location=(c.x, c.y, c.z + 0.6))
        cyc = bpy.context.object
        cyc.data.materials.append(S._principled("backdrop", (0.32, 0.27, 0.22), rough=0.9))
        for p in cyc.data.polygons:
            p.use_smooth = True
        # the camera: three-quarter view from the wrestler's front, a little above
        yaw = math.atan2(*reversed(Vector(D["dyn_mat"][k, gi].reshape(3, 3)[:, 0].tolist()).xy[:]))
        az = yaw + math.pi + math.radians(30)        # the front (the display) is -x
        # PORTRAIT_TURN_DEG (10/23): the hero3 dump's ceremony frame came out from behind (the
        # ports, not the display): 180 turns the camera round to the face
        az += math.radians(float(os.environ.get("PORTRAIT_TURN_DEG", "0")))
        dist = 1.05
        cam = bpy.data.cameras.new("cam")
        cam.lens = 85
        cam.dof.use_dof = True
        cam.dof.focus_distance = dist
        cam.dof.aperture_fstop = 4.0
        co = bpy.data.objects.new("cam", cam)
        sc.collection.objects.link(co)
        co.location = c + Vector((math.cos(az) * dist, math.sin(az) * dist, 0.30))
        look = c + Vector((0, 0, -0.03))
        co.rotation_mode = "QUATERNION"
        co.rotation_quaternion = (look - co.location).to_track_quat("-Z", "Y")
        sc.camera = co
        # lights: a large soft key, a fill, a rim
        for nm, e, size, off in (("key", 60, 0.8, (0.5, -0.4, 0.6)), ("fill", 15, 1.0, (-0.6, -0.2, 0.3)),
                                 ("rim", 40, 0.4, (-0.2, 0.6, 0.5))):
            L = bpy.data.lights.new(nm, "AREA")
            L.energy, L.size = e, size
            lo = bpy.data.objects.new(nm, L)
            sc.collection.objects.link(lo)
            lo.location = c + Vector((off[0] * math.cos(az) - off[1] * math.sin(az),
                                      off[0] * math.sin(az) + off[1] * math.cos(az), off[2]))
            lo.rotation_mode = "QUATERNION"
            lo.rotation_quaternion = (c - lo.location).to_track_quat("-Z", "Y")
        S.world(0.15, (1.0, 0.95, 0.9))
        S.setup_render("eevee", 64, "720x960", os.path.join(out, f"portrait_{ent}_{k}.png"))
        sc.render.film_transparent = False
        bpy.ops.render.render(write_still=True)
        print(f"[portrait] {ent} frame {k} -> {sc.render.filepath}")
