"""A sumo bout rebuilt in Blender from a MuJoCo replay (10/19): the robots as the replay
moved them (tools/viewer/blender_dump.py), in a hall made here in detail, rendered on
the PC's GPU. Run with Blender 5.2:

    blender -b -P tools/blender/sumo_scene.py -- --dump bout.npz --venue arena \
        --out frames/ [--frames 0:300] [--still 120] [--engine eevee|cycles] \
        [--samples 64] [--res 1920x1080] [--save scene.blend]

Venues (--venue):
    arena      a large hall in the manner of a grand tournament, made up (no real venue
               copied, no names, marks or real people -- the user, 10/19): the MuJoCo
               hall's seats, walls and crowd brought over with real materials, the
               dohyo and the hanging roof made again in detail
    shrine     a hono-zumo ring at a made-up shrine: outdoors, a roofed dohyo on gravel,
               cedars, a torii, stone lanterns, daylight
    keikoba    a stable's practice room: the ring in a wooden floor, a raised tatami
               edge, a kamidana, light through the windows

The dohyo follows tasks/jumper/sumo/env_cfg.py: a real one scaled by k = 0.8 / 2.275
(ring 4.55 m across -> 1.6 m), the mound 6.7 m square and 45 cm high, bales 14 cm thick
showing 4 cm, 20 shobu-dawara with the four tokudawara set out by a bale, 7 kaku-dawara a
side, the white shikiri lines 90 cm long 70 cm apart. Shomen is -y, east -x.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

# ── the dohyo's figures (env_cfg.py) ───────────────────────────────────────────
R = 0.8                      # ring radius
K = R / 2.275                # scale of a real dohyo
P_HALF = 6.7 / 2 * K         # mound half width
P_H = 0.45 * K               # mound height (the clay top is z = 0, the floor -P_H)
BALE_R = 0.07 * K
BALE_SHOW = 0.04 * K
SAND_W = 0.25 * K
RUN = 0.30 * K
LINE_L, LINE_W, LINE_GAP = 0.90 * K, 0.06 * K, 0.70 * K
FLOOR = -P_H

# The four tassels by corner (x sign, y sign), as audience.py: blue shomen-east,
# black shomen-west, red muko-east, white muko-west.
TASSELS = {(-1, -1): (0.12, 0.40, 0.42), (1, -1): (0.06, 0.06, 0.07),
           (-1, 1): (0.78, 0.12, 0.10), (1, 1): (0.93, 0.93, 0.90)}
ROOF_EAVE = 1.80             # audience.py _roof
ROOF_W = P_HALF * 1.02

# MuJoCo hall colours (audience.py) -> material names
HALL_COLOURS = {
    "tamari": (0.50, 0.12, 0.14), "masu_cushion": (0.66, 0.24, 0.12), "wood": (0.42, 0.30, 0.20),
    "dark_wood": (0.20, 0.14, 0.10), "mat": (0.66, 0.62, 0.44), "rail": (0.78, 0.78, 0.80),
    "chair": (0.55, 0.10, 0.12), "gold": (0.80, 0.64, 0.26), "maku": (0.36, 0.14, 0.48),
    "wall": (0.16, 0.11, 0.08), "judge_cushion": (0.42, 0.16, 0.50),
}
ROOF_NAMES = ("roof_", "katsuogi", "chigi", "mizuhiki", "eave_trim", "fusa", "roof_cable")


def srgb(c):
    """MuJoCo colours are display colours; Blender's base colour is linear."""
    return tuple(float(x) ** 2.2 if x > 0.04045 else float(x) / 12.92 for x in c[:3])


# ── materials ─────────────────────────────────────────────────────────────────
_MATS: dict = {}


def _principled(name, colour, rough=0.5, metal=0.0, coat=0.0, sheen=0.0, linear=False):
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*(colour if linear else srgb(colour)), 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if "Coat Weight" in b.inputs:
        b.inputs["Coat Weight"].default_value = coat
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = sheen
    _MATS[name] = m
    return m


def _textured(name, c1, c2, scale, detail=6.0, rough=0.8, bump=0.3, kind="noise", wave_dir="X",
              mapping_scale=(1, 1, 1), bump_dist=0.002, stripes=None, rings=None):
    """A procedural material: two colours mixed by a noise (or wave) texture in object
    coordinates, the same texture as a bump. `stripes` (count, darkness): bands along
    x, as a straw bale's rope bindings."""
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = mapping_scale
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    if kind == "wave":
        tex = N.new("ShaderNodeTexWave")
        tex.bands_direction = wave_dir
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Distortion"].default_value = 6.0
        tex.inputs["Detail"].default_value = detail
        fac = tex.outputs["Fac"]
    elif kind == "voronoi":
        tex = N.new("ShaderNodeTexVoronoi")
        tex.inputs["Scale"].default_value = scale
        fac = tex.outputs["Distance"]
    else:
        tex = N.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = scale
        tex.inputs["Detail"].default_value = detail
        fac = tex.outputs["Fac"]
    L.new(mp.outputs["Vector"], tex.inputs["Vector"])
    ramp = N.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*srgb(c1), 1)
    ramp.color_ramp.elements[1].color = (*srgb(c2), 1)
    L.new(fac, ramp.inputs["Fac"])
    col = ramp.outputs["Color"]
    if stripes:
        count, dark = stripes
        sep = N.new("ShaderNodeSeparateXYZ")
        L.new(tc.outputs["Object"], sep.inputs["Vector"])
        mul = N.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = count
        L.new(sep.outputs["X"], mul.inputs[0])
        fr = N.new("ShaderNodeMath"); fr.operation = "FRACT"
        L.new(mul.outputs[0], fr.inputs[0])
        band = N.new("ShaderNodeMath"); band.operation = "LESS_THAN"; band.inputs[1].default_value = 0.12
        L.new(fr.outputs[0], band.inputs[0])
        mix = N.new("ShaderNodeMix"); mix.data_type = "RGBA"
        L.new(band.outputs[0], mix.inputs["Factor"])
        L.new(col, mix.inputs[6])
        mix.inputs[7].default_value = (*srgb([c * dark for c in c1]), 1)
        col = mix.outputs[2]
    L.new(col, b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    bp = N.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = bump
    bp.inputs["Distance"].default_value = bump_dist
    L.new(fac, bp.inputs["Height"])
    normal = bp.outputs["Normal"]
    if rings:
        # hakime: the broom's strokes round the ring's centre, faint concentric grooves
        count, strength = rings
        rw = N.new("ShaderNodeTexWave")
        rw.wave_type = "RINGS"
        try:
            rw.rings_direction = "Z"
        except Exception:
            pass
        rw.inputs["Scale"].default_value = count
        rw.inputs["Distortion"].default_value = 2.5
        rw.inputs["Detail"].default_value = 3.0
        L.new(tc.outputs["Object"], rw.inputs["Vector"])
        b2 = N.new("ShaderNodeBump")
        b2.inputs["Strength"].default_value = strength
        b2.inputs["Distance"].default_value = 0.0008
        L.new(rw.outputs["Fac"], b2.inputs["Height"])
        L.new(normal, b2.inputs["Normal"])
        normal = b2.outputs["Normal"]
    L.new(normal, b.inputs["Normal"])
    _MATS[name] = m
    return m


def _floorboards(name, c1, c2, plank=(1.8, 0.16), rough=0.45):
    """Floor planks: a brick texture with thin dark seams, the grain a stretched noise."""
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    br = N.new("ShaderNodeTexBrick")
    br.offset = 0.5
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Brick Width"].default_value = plank[0]
    br.inputs["Row Height"].default_value = plank[1]
    br.inputs["Mortar Size"].default_value = 0.003
    br.inputs["Color1"].default_value = (*srgb(c1), 1)
    br.inputs["Color2"].default_value = (*srgb(c2), 1)
    br.inputs["Mortar"].default_value = (*srgb([c * 0.35 for c in c1]), 1)
    L.new(tc.outputs["Object"], br.inputs["Vector"])
    grain = N.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 30
    grain.inputs["Detail"].default_value = 8
    mp = N.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (0.08, 1.0, 1.0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    L.new(mp.outputs["Vector"], grain.inputs["Vector"])
    mix = N.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.35
    L.new(br.outputs["Color"], mix.inputs[6])
    L.new(grain.outputs["Color"], mix.inputs[7])
    L.new(mix.outputs[2], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = rough
    bp = N.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.3
    bp.inputs["Distance"].default_value = 0.002
    L.new(br.outputs["Fac"], bp.inputs["Height"])
    L.new(bp.outputs["Normal"], b.inputs["Normal"])
    _MATS[name] = m
    return m


def _emission(name, colour, strength):
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Emission Color"].default_value = (*srgb(colour), 1)
    b.inputs["Emission Strength"].default_value = strength
    b.inputs["Base Color"].default_value = (*srgb(colour), 1)
    _MATS[name] = m
    return m


def mats():
    clay = (0.62, 0.43, 0.28)
    return dict(
        clay=_textured("clay", (0.47, 0.30, 0.18), (0.60, 0.40, 0.25), 60, rough=0.92, bump=0.15,
                       bump_dist=0.0015, rings=(55.0, 0.25)),
        clay_side=_textured("clay_side", (0.47, 0.32, 0.20), (0.60, 0.42, 0.27), 90, detail=8, rough=0.95,
                            bump=0.25, bump_dist=0.0015),
        sand=_textured("sand", (0.60, 0.51, 0.37), (0.74, 0.66, 0.50), 900, rough=0.97, bump=0.5,
                       bump_dist=0.0008),
        straw=_textured("straw", (0.56, 0.45, 0.24), (0.72, 0.60, 0.36), 140, kind="wave", wave_dir="Z",
                        rough=0.85, bump=0.6, bump_dist=0.0012, mapping_scale=(0.15, 1, 1),
                        stripes=(28.0, 0.55)),
        line=_principled("line", (0.95, 0.95, 0.92), rough=0.6),
        wood=_textured("wood", (0.36, 0.24, 0.14), (0.50, 0.35, 0.21), 18, kind="wave", wave_dir="X",
                       rough=0.55, bump=0.08),
        dark_wood=_textured("dark_wood", (0.12, 0.08, 0.05), (0.24, 0.16, 0.10), 18, kind="wave",
                            wave_dir="X", rough=0.45, bump=0.06),
        roof=_textured("roof_wood", (0.055, 0.04, 0.03), (0.12, 0.085, 0.06), 30, kind="wave", wave_dir="Y",
                       rough=0.45, bump=0.15),
        gold=_principled("gold", (0.85, 0.66, 0.30), rough=0.28, metal=1.0),
        maku=_textured("maku", (0.30, 0.10, 0.42), (0.42, 0.17, 0.55), 6, rough=0.75, bump=0.05),
        white_cloth=_principled("white_cloth", (0.93, 0.92, 0.88), rough=0.85, sheen=0.4),
        cushion=_principled("cushion", HALL_COLOURS["masu_cushion"], rough=0.8, sheen=0.6),
        tamari=_principled("tamari", HALL_COLOURS["tamari"], rough=0.8, sheen=0.6),
        judge_cushion=_principled("judge_cushion", HALL_COLOURS["judge_cushion"], rough=0.8, sheen=0.6),
        mat=_textured("tatami", (0.58, 0.55, 0.36), (0.72, 0.68, 0.46), 260, kind="wave", wave_dir="Y",
                      rough=0.8, bump=0.25, bump_dist=0.0006),
        rail=_principled("rail", (0.80, 0.80, 0.82), rough=0.25, metal=1.0),
        chair=_principled("chair", HALL_COLOURS["chair"], rough=0.55, sheen=0.3),
        wall=_textured("wall", (0.12, 0.08, 0.06), (0.20, 0.13, 0.09), 8, kind="wave", wave_dir="Z",
                       rough=0.6, bump=0.05),
        floor=_textured("hall_floor", (0.09, 0.07, 0.06), (0.15, 0.11, 0.09), 25, rough=0.5, bump=0.05),
        gravel=_textured("gravel", (0.22, 0.20, 0.17), (0.40, 0.37, 0.32), 420, detail=10, rough=0.95,
                         bump=0.9, bump_dist=0.003),
        stone=_textured("stone", (0.38, 0.38, 0.36), (0.55, 0.54, 0.50), 25, rough=0.9, bump=0.5),
        cedar=_textured("cedar", (0.06, 0.15, 0.06), (0.12, 0.25, 0.10), 40, rough=0.8, bump=0.4),
        bark=_textured("bark", (0.20, 0.12, 0.07), (0.32, 0.20, 0.12), 30, kind="wave", wave_dir="Z",
                       rough=0.9, bump=0.6),
        vermilion=_principled("vermilion", (0.80, 0.20, 0.10), rough=0.5),
        black=_principled("lacquer_black", (0.03, 0.03, 0.03), rough=0.25, coat=0.6),
        thatch=_textured("thatch", (0.30, 0.24, 0.16), (0.46, 0.38, 0.25), 120, kind="wave", wave_dir="Y",
                         rough=0.95, bump=0.7),
        paper=_emission("lantern_paper", (1.0, 0.85, 0.6), 3.0),
        plaster=_principled("plaster", (0.86, 0.83, 0.76), rough=0.9),
    )


# ── mesh helpers ──────────────────────────────────────────────────────────────


def _new_mesh_obj(name, verts, faces, mat=None, smooth=False, collection=None, validate=True):
    me = bpy.data.meshes.new(name)
    v = np.asarray(verts, np.float32).reshape(-1, 3)
    f = np.asarray(faces, np.int32).reshape(-1, 3)
    me.vertices.add(len(v))
    me.vertices.foreach_set("co", v.ravel())
    me.loops.add(len(f) * 3)
    me.loops.foreach_set("vertex_index", f.ravel())
    me.polygons.add(len(f))
    me.polygons.foreach_set("loop_start", np.arange(0, len(f) * 3, 3, dtype=np.int32))
    me.update(calc_edges=True)
    if validate:
        me.validate(clean_customdata=False)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if mat is not None:
        me.materials.append(mat)
    return ob


def _add(op, name, mat=None, smooth=False, **kw):
    op(**kw)
    ob = bpy.context.active_object
    ob.name = name
    if mat is not None:
        ob.data.materials.append(mat)
    if smooth:
        bpy.ops.object.shade_smooth()
    return ob


def box(name, center, half, mat, rot=(0, 0, 0), bevel=0.0, apply=False):
    ob = _add(bpy.ops.mesh.primitive_cube_add, name, mat, size=2, location=center, rotation=rot)
    ob.scale = half
    if apply:
        # textures in metres (object coordinates of the scaled box, not of the unit cube)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        m = ob.modifiers.new("bevel", "BEVEL")
        m.width = bevel
        m.segments = 2
        m.limit_method = "NONE"
    return ob


def cyl(name, center, r, depth, mat, rot=(0, 0, 0), verts=24, smooth=True):
    return _add(bpy.ops.mesh.primitive_cylinder_add, name, mat, smooth, vertices=verts, radius=r, depth=depth,
                location=center, rotation=rot)


def _prim_tris(gtype, size):
    """A MuJoCo primitive as triangles in its own frame."""
    if gtype == 6:                                            # box
        x, y, z = size
        v = np.array([[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        return v, np.array(f)
    if gtype == 0:                                            # plane
        x, y = size[0] or 50, size[1] or 50
        return np.array([[-x, -y, 0], [x, -y, 0], [x, y, 0], [-x, y, 0]]), np.array([[0, 1, 2], [0, 2, 3]])
    n = 20
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if gtype == 5:                                            # cylinder
        r, h = size[0], size[1]
        ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        v = np.concatenate([np.c_[ring, -h * np.ones(n)], np.c_[ring, h * np.ones(n)], [[0, 0, -h], [0, 0, h]]])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        return v, np.array(f)
    # sphere / ellipsoid / capsule: a UV sphere, stretched
    m = 10
    lat = np.linspace(-np.pi / 2, np.pi / 2, m + 1)
    v, f = [], []
    for la in lat:
        for lo in a:
            v.append([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    v = np.array(v)
    for i in range(m):
        for k in range(n):
            p, q = i * n + k, i * n + (k + 1) % n
            f += [[p, q, q + n], [p, q + n, p + n]]
    if gtype == 4:
        v = v * np.asarray(size[:3])
    elif gtype == 3:                                          # capsule: radius, half length
        r, h = size[0], size[1]
        v = v * r
        v[:, 2] += np.sign(v[:, 2]) * h
    else:
        v = v * size[0]
    return v, np.array(f)


#: EEVEE exposure per venue (the shrine's sky is bright outdoors).
EXPOSURE_EEVEE = {"arena": -0.6, "shrine": -0.6, "keikoba": -0.6}
# the look (--look): 3 as rendered on 10/20 (M0094 on), 4 the crowd muted and the hall dimmer
LOOK = 3
CROWD_COLOURS = ""     # --crowd-colours: a crowd_figures.py file (one colour per spectator)
PORTRAITS = ""         # --portraits: portraits.py's pictures for the hall's portrait boxes
CROWD_CELL = None      # --crowd-cell: the crowd's clustering cell (m); 0 keeps the meshes (look 5's default)


# ── the dohyo ─────────────────────────────────────────────────────────────────


def _bale_mesh(name, length, radius, curve_r=None, mat=None):
    """A straw bale: a tube, rounded at its ends, straight or bent round the ring's
    centre (curve_r: the radius of the bend, the bale along +x at y = curve_r)."""
    nu, nv = 40, 16
    v, f = [], []
    for i in range(nu + 1):
        t = -0.5 + i / nu
        # ends pinched: a bale is tied tight at its ends
        pinch = 1.0 - 0.35 * max(0.0, abs(t) * 2 - 0.86) / 0.14
        for j in range(nv):
            a = 2 * math.pi * j / nv
            rr = radius * pinch
            x, y, z = t * length, rr * math.cos(a), rr * math.sin(a)
            if curve_r:
                ang = x / curve_r
                rad = curve_r + y
                x, y = rad * math.sin(ang), rad * math.cos(ang) - curve_r
            v.append((x, y, z))
    for i in range(nu):
        for j in range(nv):
            p, q = i * nv + j, i * nv + (j + 1) % nv
            f += [(p, q, q + nv), (p, q + nv, p + nv)]
    c0, c1 = len(v), len(v) + 1
    v += [v[0][:1] + (0, 0) if not curve_r else (0, 0, 0), (0, 0, 0)]
    # caps
    v[c0] = tuple(np.mean(v[:nv], 0))
    v[c1] = tuple(np.mean(v[nu * nv:nu * nv + nv], 0))
    for j in range(nv):
        f.append((c0, (j + 1) % nv, j))
        f.append((c1, nu * nv + j, nu * nv + (j + 1) % nv))
    return _new_mesh_obj(name, v, f, mat, smooth=True)


def build_dohyo(M, sunk=False):
    """The mound (or, sunk, a ring in a floor), its bales, sand, lines, salt and water."""
    col = bpy.data.collections.new("dohyo")
    bpy.context.scene.collection.children.link(col)
    top, bot = P_HALF, P_HALF + RUN
    if not sunk:
        v = [(sx * top, sy * top, 0) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        v += [(sx * bot, sy * bot, FLOOR) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        f = [(0, 1, 2), (0, 2, 3)]
        for i in range(4):
            j = (i + 1) % 4
            f += [(i, 4 + i, 4 + j), (i, 4 + j, j)]
        mound = _new_mesh_obj("mound", v, f, None)
        mound.data.materials.append(M["clay"])
        mound.data.materials.append(M["clay_side"])
        for p in mound.data.polygons:
            p.material_index = 0 if p.index < 2 else 1
        sub = mound.modifiers.new("detail", "SUBSURF")
        sub.subdivision_type = "SIMPLE"
        sub.levels = sub.render_levels = 5
        disp = mound.modifiers.new("trodden", "DISPLACE")
        tex = bpy.data.textures.new("clay_lumps", "CLOUDS")
        tex.noise_scale = 0.05
        disp.texture = tex
        disp.strength = 0.0012
    # janome sand and the clay inside the ring
    jan = cyl("janome_sand", (0, 0, 0.0003), R + BALE_R + SAND_W, 0.0006, M["sand"], verts=128)
    clay_in = cyl("ring_clay", (0, 0, 0.0007), R - BALE_R, 0.0006, M["clay"], verts=128)
    # shobu-dawara: 20 slots, the four on the axes set out by a bale (tokudawara)
    slots = 20
    for i in range(slots):
        a = 2 * math.pi * (i + 0.5) / slots
        on_axis = any(abs(math.remainder(a - q * math.pi / 2, 2 * math.pi)) < math.pi / slots
                      for q in range(4))
        r = R + (2 * BALE_R if on_axis else 0.0)
        ob = _bale_mesh(f"shobu_dawara_{i}", r * 2 * math.pi / slots * 0.95, BALE_R, curve_r=r, mat=M["straw"])
        ob.location = (0, 0, -BALE_R + BALE_SHOW)
        ob.rotation_euler = (0, 0, a - math.pi / 2)
        # the bale sits on the ring at angle a: move its bend centre to the origin
        ob.data.transform(Matrix.Translation((0, r, 0)))
    if not sunk:
        # kaku-dawara along the mound's top edge, 7 a side
        edge = top - BALE_R * 1.2
        for side, (ux, uy, nx, ny) in enumerate(((1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, 1, -1, 0))):
            for k in range(7):
                t = -edge + (k + 0.5) * 2 * edge / 7
                ob = _bale_mesh(f"kaku_dawara_{side}_{k}", 2 * edge / 7 * 0.96, BALE_R, mat=M["straw"])
                ob.location = (ux * t + nx * edge, uy * t + ny * edge, -BALE_R + BALE_SHOW)
                ob.rotation_euler = (0, 0, math.atan2(uy, ux))
        # age-dawara at the corners, fumi-dawara steps up three sides
        for sx in (-1, 1):
            for sy in (-1, 1):
                for k in range(2):
                    ob = _bale_mesh(f"age_dawara_{sx}_{sy}_{k}", 0.16, BALE_R, mat=M["straw"])
                    ob.location = (sx * (top + 0.02 + 0.05 * k), sy * (top - 0.12), -BALE_R + BALE_SHOW - 0.03 * k)
                    ob.rotation_euler = (0, 0, math.pi / 2)
        for side, (ox, oy) in enumerate(((-1, 0), (1, 0), (0, 1))):
            for k in range(3):
                d = top + RUN * (k + 0.5) / 3
                z = FLOOR * (k + 0.5) / 3
                ob = _bale_mesh(f"fumi_dawara_{side}_{k}", 0.22, BALE_R * 0.9, mat=M["straw"])
                ob.location = (ox * d, oy * d, z + BALE_R * 0.4)
                ob.rotation_euler = (0, 0, 0 if ox == 0 else math.pi / 2)
    for sign in (-1, 1):
        box(f"shikiri_{'e' if sign < 0 else 'w'}", (sign * (LINE_GAP + LINE_W) / 2, 0, 0.0013),
            (LINE_W / 2, LINE_L / 2, 0.0004), M["line"])
    # salt boxes and water buckets at the east and west corners on the shomen side
    for sx in (-1, 1):
        x, y = sx * (top - 0.10), -(top - 0.10)
        box(f"salt_box_{sx}", (x, y, 0.035), (0.05, 0.035, 0.035), M["wood"], bevel=0.004)
        _add(bpy.ops.mesh.primitive_ico_sphere_add, f"salt_{sx}", M["white_cloth"], True,
             subdivisions=3, radius=0.03, location=(x, y, 0.07)).scale = (1.4, 1, 0.45)
        cyl(f"chikara_mizu_{sx}", (x, y + 0.14, 0.045), 0.035, 0.09, M["wood"])
        cyl(f"chikara_mizu_band_{sx}", (x, y + 0.14, 0.06), 0.0365, 0.008, M["rail"])
    for ob in list(bpy.context.scene.collection.objects):
        if ob.name.split("_")[0] in ("mound", "janome", "ring", "shobu", "kaku", "age", "fumi", "shikiri",
                                      "salt", "chikara"):
            bpy.context.scene.collection.objects.unlink(ob)
            col.objects.link(ob)


def build_roof(M, eave=ROOF_EAVE, w=ROOF_W, hung=True):
    """A shinmei-style roof: two slopes, a ridge with katsuogi, chigi crossed at the
    gables, the purple mizuhiki-maku round the eaves with its scalloped hem, a tassel at
    each corner, and (hung) the cables it hangs from."""
    pitch = math.radians(28)
    run = w * 1.08
    rise = run * math.tan(pitch)
    length = w * 1.12
    for s in (-1, 1):
        ob = box(f"roof_slope_{s}", (0, -s * run / 2, eave + rise / 2), (length, run / math.cos(pitch) / 2, 0.03),
                 M["roof"], rot=(s * pitch, 0, 0), bevel=0.006)
        # a layered thatch edge under each slope
        box(f"roof_layer_{s}", (0, -s * run * 0.52, eave + rise / 2 - 0.035), (length * 1.01, run / math.cos(pitch) / 2 * 1.02, 0.008),
            M["dark_wood"], rot=(s * pitch, 0, 0))
    ridge = eave + rise + 0.02
    box("roof_ridge", (0, 0, ridge), (length * 1.02, 0.05, 0.045), M["roof"], bevel=0.008)
    box("roof_ridge_cap", (0, 0, ridge + 0.05), (length * 1.03, 0.035, 0.012), M["gold"])
    for i in range(7):
        x = -w * 0.9 + i * (1.8 * w / 6)
        cyl(f"katsuogi_{i}", (x, 0, ridge + 0.075), 0.032, 0.20, M["roof"], rot=(math.pi / 2, 0, 0))
        for e in (-1, 1):
            cyl(f"katsuogi_cap_{i}_{e}", (x, e * 0.1, ridge + 0.075), 0.034, 0.012, M["gold"], rot=(math.pi / 2, 0, 0))
    for sx in (-1, 1):
        for s in (-1, 1):
            a = s * math.radians(35)
            box(f"chigi_{sx}_{s}", (sx * length, s * 0.07, ridge + 0.09), (0.012, 0.03, 0.24), M["roof"],
                rot=(a, 0, 0), bevel=0.003)
    # mizuhiki-maku: a curtain round the eaves, scalloped (three swags a side), with crests
    for k in range(4):
        ang = k * math.pi / 2
        out = np.array([math.cos(ang), math.sin(ang)])
        along = np.array([-math.sin(ang), math.cos(ang)])
        nu, nv = 120, 10
        v, f = [], []
        for i in range(nu + 1):
            t = -w + 2 * w * i / nu
            swag = 0.035 * (1 - math.cos(2 * math.pi * 3 * (t + w) / (2 * w))) / 2
            fold = 0.006 * math.sin(2 * math.pi * 40 * (t + w) / (2 * w))
            for j in range(nv + 1):
                h = j / nv
                depth = 0.20 + swag
                z = eave + 0.02 - h * depth
                p = out * (w + fold * h) + along * t
                v.append((p[0], p[1], z))
        for i in range(nu):
            for j in range(nv):
                p, q = i * (nv + 1) + j, (i + 1) * (nv + 1) + j
                f += [(p, q, q + 1), (p, q + 1, p + 1)]
        ob = _new_mesh_obj(f"mizuhiki_{k}", v, f, M["maku"], smooth=True)
        sol = ob.modifiers.new("cloth", "SOLIDIFY")
        sol.thickness = 0.004
        for c in (-0.66, 0.0, 0.66):
            p = out * (w + 0.008) + along * (c * w)
            cyl(f"maku_crest_{k}_{c}", (p[0], p[1], eave - 0.07), 0.035, 0.002, M["white_cloth"],
                rot=(math.pi / 2, 0, ang + math.pi / 2), verts=32)
        box(f"eave_trim_{k}", (*(out * w), eave + 0.035), (0.012, w + 0.01, 0.022), M["gold"],
            rot=(0, 0, ang))
    for (sx, sy), c in TASSELS.items():
        x, y = sx * w, sy * w
        mat = _principled(f"fusa_{sx}_{sy}", c, rough=0.7, sheen=0.8)
        _add(bpy.ops.mesh.primitive_uv_sphere_add, f"fusa_knot_{sx}_{sy}", mat, True, radius=0.07,
             location=(x, y, eave - 0.24))
        cyl(f"fusa_cord_{sx}_{sy}", (x, y, eave - 0.12), 0.012, 0.22, mat)
        # the tassel's fringe: many strands
        # a real fusa is about 2.3 m long and full: scaled, a fringe 8 cm wide at the knot
        # opening to 16 cm, 0.5 m long
        strands = 180
        v, fcs = [], []
        for i in range(strands):
            a = 2 * math.pi * i / strands * 7.3
            rr = 0.02 + 0.06 * ((i * 37) % strands) / strands
            top_ = (x + rr * math.cos(a) * 0.55, y + rr * math.sin(a) * 0.55, eave - 0.29)
            bot_ = (x + rr * math.cos(a) * 1.3, y + rr * math.sin(a) * 1.3, eave - 0.80 - 0.03 * ((i * 7) % 5) / 4)
            n0 = len(v)
            d = 0.006
            v += [top_, (top_[0] + d, top_[1], top_[2]), bot_, (bot_[0] + d, bot_[1], bot_[2])]
            fcs += [(n0, n0 + 1, n0 + 3), (n0, n0 + 3, n0 + 2)]
        _new_mesh_obj(f"fusa_{sx}_{sy}", v, fcs, mat, smooth=True)
    if hung:
        for sx in (-1, 1):
            for sy in (-1, 1):
                cyl(f"roof_cable_{sx}_{sy}", (sx * w * 0.6, sy * w * 0.6, ridge + 1.5), 0.004, 3.0, M["rail"])


# ── look 5: the hall's furnishings in detail ───────────────────────────────────
# 10/20 (the user: the decorations at the real thing's level, not the ring alone): the salt
# basket on its stand and the water bucket with its ladle and the paper beside it, the roof
# with its gable boards, rafters and gold fittings, the copper roof's seams, a faint haze in
# the hall's light. Generic forms of the tradition only: no real venue's marks or names.


def _woven(name, c1, c2):
    """A woven basket's surface: a checker of bands, bumped."""
    if name in _MATS:
        return _MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    N, L = nt.nodes, nt.links
    b = N["Principled BSDF"]
    tc = N.new("ShaderNodeTexCoord")
    ch = N.new("ShaderNodeTexChecker")
    ch.inputs["Scale"].default_value = 220
    ch.inputs["Color1"].default_value = (*srgb(c1), 1)
    ch.inputs["Color2"].default_value = (*srgb(c2), 1)
    L.new(tc.outputs["Object"], ch.inputs["Vector"])
    L.new(ch.outputs["Color"], b.inputs["Base Color"])
    bp = N.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.6
    bp.inputs["Distance"].default_value = 0.0008
    L.new(ch.outputs["Fac"], bp.inputs["Height"])
    L.new(bp.outputs["Normal"], b.inputs["Normal"])
    b.inputs["Roughness"].default_value = 0.8
    _MATS[name] = m
    return m


def _lathe(name, profile, mat, segs=48, smooth=True):
    """A surface of revolution about z: profile [(r, z), ...] from the bottom up."""
    v, f = [], []
    n = len(profile)
    for i in range(segs):
        a = 2 * math.pi * i / segs
        for r_, z in profile:
            v.append((r_ * math.cos(a), r_ * math.sin(a), z))
    for i in range(segs):
        i2 = (i + 1) % segs
        for j in range(n - 1):
            p, q = i * n + j, i2 * n + j
            f += [(p, q, q + 1), (p, q + 1, p + 1)]
    c0 = len(v)
    v.append((0, 0, profile[0][1]))
    for i in range(segs):
        f.append((c0, ((i + 1) % segs) * n, i * n))
    return _new_mesh_obj(name, v, f, mat, smooth=smooth)


def detail_ring_props(M):
    """The salt and the water at the mound's shomen corners, as a ring keeps them: salt
    heaped in a bamboo basket on a black lacquered stand, a wooden bucket with copper hoops
    and a ladle across it, and a stack of white paper (chikara-gami) on a tray."""
    top = P_HALF
    basket = _woven("bamboo", (0.52, 0.40, 0.22), (0.66, 0.53, 0.32))
    copper = _principled("copper", (0.62, 0.36, 0.20), rough=0.35, metal=1.0)
    for ob in list(bpy.data.objects):
        if ob.name.startswith(("salt_box_", "salt_", "chikara_mizu_")):
            bpy.data.objects.remove(ob)
    for sx in (-1, 1):
        x, y = sx * (top - 0.10), -(top - 0.10)
        box(f"salt_stand_{sx}", (x, y, 0.028), (0.042, 0.042, 0.004), M["black"], bevel=0.002)
        for lx in (-1, 1):
            for ly in (-1, 1):
                box(f"salt_leg_{sx}_{lx}_{ly}", (x + lx * 0.036, y + ly * 0.036, 0.013), (0.004, 0.004, 0.013),
                    M["black"])
        bk = _lathe(f"salt_basket_{sx}", [(0.026, 0.0), (0.034, 0.006), (0.042, 0.022), (0.044, 0.030),
                                          (0.040, 0.030), (0.038, 0.024)], basket)
        bk.location = (x, y, 0.032)
        heap = _lathe(f"salt_heap_{sx}", [(0.040, 0.0), (0.030, 0.010), (0.016, 0.018), (0.0, 0.021)],
                      M["white_cloth"])
        heap.location = (x, y, 0.058)
        bx, by = x, y + 0.14
        tub = _lathe(f"chikara_oke_{sx}", [(0.030, 0.0), (0.034, 0.055), (0.035, 0.060), (0.032, 0.060),
                                           (0.031, 0.006)], M["wood"])
        tub.location = (bx, by, 0.0)
        for hz in (0.012, 0.046):
            cyl(f"chikara_hoop_{sx}_{hz}", (bx, by, hz), 0.0335 + 0.004 * hz / 0.06, 0.004, copper)
        cyl(f"chikara_water_{sx}", (bx, by, 0.052), 0.031, 0.001,
            _principled("water", (0.25, 0.28, 0.27), rough=0.02, coat=1.0))
        cyl(f"hishaku_handle_{sx}", (bx, by, 0.064), 0.0025, 0.11, M["wood"], rot=(0, math.pi / 2 - 0.12, 0.6))
        cup = _lathe(f"hishaku_cup_{sx}", [(0.0, 0.0), (0.012, 0.0), (0.013, 0.016), (0.011, 0.016)], M["wood"])
        cup.location = (bx + 0.045 * math.cos(0.6), by + 0.045 * math.sin(0.6), 0.06)
        px, py = x, y + 0.075
        box(f"kami_tray_{sx}", (px, py, 0.004), (0.028, 0.02, 0.004), M["black"], bevel=0.001)
        for k in range(6):
            box(f"kami_{sx}_{k}", (px + 0.001 * k, py, 0.009 + 0.0012 * k), (0.024, 0.017, 0.0005),
                M["white_cloth"], rot=(0, 0, 0.03 * (k - 3)))


def detail_roof(M, eave=ROOF_EAVE, w=ROOF_W):
    """The roof as built: rafters showing under the eaves, black gable boards with gold
    six-leaf studs and a gold pendant at each gable, the copper sheets' standing seams."""
    pitch = math.radians(28)
    run = w * 1.08
    rise = run * math.tan(pitch)
    length = w * 1.12
    for s in (-1, 1):
        for i in range(40):
            x = -length + (i + 0.5) * 2 * length / 40
            box(f"taruki_{s}_{i}", (x, -s * (run * 0.98), eave + 0.01), (0.006, 0.05, 0.008), M["dark_wood"],
                rot=(s * pitch, 0, 0))
    ridge = eave + rise + 0.02
    for sx in (-1, 1):
        cx = sx * (length + 0.012)
        for s in (-1, 1):
            box(f"hafu_{sx}_{s}", (cx, -s * run / 2, eave + rise / 2 + 0.01),
                (0.008, run / math.cos(pitch) / 2 + 0.02, 0.035), M["black"], rot=(s * pitch, 0, 0), bevel=0.003)
            for k in range(5):
                t = (k + 0.5) / 5
                y = -s * run * (1 - t)
                z = eave + rise * t + 0.012
                cyl(f"rokuyo_{sx}_{s}_{k}", (cx + sx * 0.009, y, z), 0.011, 0.004, M["gold"],
                    rot=(0, math.pi / 2, 0), verts=6)
        gx = sx * (length + 0.022)
        cyl(f"gegyo_{sx}", (gx, 0, ridge - 0.07), 0.04, 0.006, M["gold"], rot=(0, math.pi / 2, 0), verts=24)
        cyl(f"gegyo_core_{sx}", (gx + sx * 0.004, 0, ridge - 0.07), 0.018, 0.006, M["black"],
            rot=(0, math.pi / 2, 0), verts=24)
    m = M["roof"]
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    if not any(n.name == "seams" for n in nt.nodes):
        tc = nt.nodes.new("ShaderNodeTexCoord")
        wv = nt.nodes.new("ShaderNodeTexWave")
        wv.name = "seams"
        wv.bands_direction = "X"
        wv.inputs["Scale"].default_value = 9.0
        wv.inputs["Distortion"].default_value = 0.0
        wv.inputs["Detail"].default_value = 0.0
        nt.links.new(tc.outputs["Object"], wv.inputs["Vector"])
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = 0.35
        bp.inputs["Distance"].default_value = 0.002
        nt.links.new(wv.outputs["Fac"], bp.inputs["Height"])
        prev = b.inputs["Normal"].links[0].from_socket if b.inputs["Normal"].links else None
        if prev is not None:
            nt.links.new(prev, bp.inputs["Normal"])
        nt.links.new(bp.outputs["Normal"], b.inputs["Normal"])


def hall_haze(density=0.012):
    """A faint haze in the hall: the ring lights' beams and the depth of the stands."""
    w = bpy.context.scene.world
    nt = w.node_tree
    vs = nt.nodes.new("ShaderNodeVolumeScatter")
    vs.inputs["Density"].default_value = density
    vs.inputs["Anisotropy"].default_value = 0.6
    out = next(n for n in nt.nodes if n.type == "OUTPUT_WORLD")
    nt.links.new(vs.outputs[0], out.inputs["Volume"])


# ── the arena: the MuJoCo hall's pieces, real materials ──────────────────────


def _material_for(rgba, M):
    c = np.asarray(rgba[:3])
    best, d = None, 1e9
    for key, ref in HALL_COLOURS.items():
        dd = float(np.sum((c - np.asarray(ref)) ** 2))
        if dd < d:
            best, d = key, dd
    if d > 0.01:
        return None
    return {"tamari": M["tamari"], "masu_cushion": M["cushion"], "wood": M["wood"], "dark_wood": M["dark_wood"],
            "mat": M["mat"], "rail": M["rail"], "chair": M["chair"], "gold": M["gold"], "maku": M["maku"],
            "wall": M["wall"], "judge_cushion": M["judge_cushion"]}[best]


def import_static(D, M, skip_dohyo=True, skip_roof=True):
    """The dump's static geoms (the hall) merged by material into a few objects."""
    groups: dict = {}
    portraits: list = []
    col = bpy.data.collections.new("hall")
    bpy.context.scene.collection.children.link(col)
    for i in range(len(D["static_type"])):
        name, body = str(D["static_name"][i]), str(D["static_body"][i])
        if skip_dohyo and not body.startswith("decor_hall"):
            continue                       # the dohyo and the floor: made here
        if skip_roof and name.startswith(ROOF_NAMES):
            continue
        gt = int(D["static_type"][i])
        if PORTRAITS and gt == 6 and np.allclose(D["static_size"][i], (0.004, 0.22, 0.30), atol=1e-4):
            portraits.append(i)            # a champion's portrait: a picture of its own below
            continue
        if gt == 7:
            mid = int(D["static_mesh"][i])
            v, f = D[f"mesh_{mid}_v"], D[f"mesh_{mid}_f"]
        else:
            v, f = _prim_tris(gt, D["static_size"][i])
        rgba = D["static_rgba"][i]
        mat = _material_for(rgba, M)
        key = mat.name if mat else tuple(np.round(rgba[:3], 2))
        R_ = D["static_mat"][i].reshape(3, 3)
        w = np.asarray(v) @ R_.T + D["static_pos"][i]
        g = groups.setdefault(key, [[], [], 0, mat, rgba])
        g[0].append(w)
        g[1].append(np.asarray(f) + g[2])
        g[2] += len(v)
    for key, (vs, fs, _, mat, rgba) in groups.items():
        if mat is None:
            mat = _principled(f"hall_{key}", rgba, rough=0.6)
        ob = _new_mesh_obj(f"hall_{getattr(mat, 'name', key)}", np.concatenate(vs), np.concatenate(fs), mat,
                           collection=col)
        bev = ob.modifiers.new("bevel", "BEVEL")
        bev.width = 0.004
        bev.segments = 1
        bev.limit_method = "ANGLE"
    print(f"[sumo_scene] hall: {sum(len(g[1]) for g in groups.values())} pieces in {len(groups)} objects")
    if portraits:
        _portraits(D, portraits, col)


def _portraits(D, ids, col):
    """The champions' portraits on the hall's walls (portraits.py's pictures of our own
    wrestlers), each a plane on the face of its MuJoCo box that looks at the ring."""
    files = sorted(Path(PORTRAITS).glob("champion_*.jpg"))
    if not files:
        print(f"[sumo_scene] no portraits in {PORTRAITS}")
        return
    hy, hz = 0.22, 0.30
    for n, i in enumerate(ids):
        img = bpy.data.images.load(str(files[n % len(files)]))
        m = bpy.data.materials.new(f"portrait_{n}")
        m.use_nodes = True
        nt = m.node_tree
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        b = nt.nodes["Principled BSDF"]
        nt.links.new(tex.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = 0.35            # behind glass
        if "Coat Weight" in b.inputs:
            b.inputs["Coat Weight"].default_value = 0.4
        R_ = D["static_mat"][i].reshape(3, 3)
        p0 = D["static_pos"][i]
        x = -0.0045                                           # the face toward the ring
        local = np.array([(x, hy, -hz), (x, -hy, -hz), (x, -hy, hz), (x, hy, hz)])
        v = local @ R_.T + p0
        # the side facing the ring is local -x: its right is local -y
        if np.dot(R_[:, 0], -np.array([p0[0], p0[1], 0.0])) > 0:   # the box turned the other way
            v = np.array([(-x, -hy, -hz), (-x, hy, -hz), (-x, hy, hz), (-x, -hy, hz)]) @ R_.T + p0
        ob = _new_mesh_obj(f"portrait_{n}", v, [(0, 1, 2), (0, 2, 3)], m, collection=col, validate=False)
        uv = ob.data.uv_layers.new(name="UV")
        uv.data.foreach_set("uv", np.array([(0, 0), (1, 0), (1, 1), (0, 0), (1, 1), (0, 1)], np.float32).ravel())
    print(f"[sumo_scene] portraits: {len(ids)} ({len(files)} pictures)")


def _cluster(v, f, c, cell):
    """Vertex clustering: vertices in one cell of `cell` metres become one (their mean),
    faces that collapse are dropped. The crowd is 21 000 figures at full detail (20 M
    triangles); seen from the ring a 10 mm cell keeps their shapes."""
    k = np.floor(v / cell).astype(np.int64) + (1 << 20)
    key = (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(uniq)
    cnt = np.bincount(inv, minlength=n).astype(np.float32)
    nv = np.stack([np.bincount(inv, weights=v[:, i], minlength=n) for i in range(3)], 1) / cnt[:, None]
    nc = np.stack([np.bincount(inv, weights=c[:, i], minlength=n) for i in range(3)], 1) / cnt[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    # each face keeps its own figure's colour (a vertex mean mixes neighbours: speckles)
    fc = c[f[keep, 0]]
    return nv.astype(np.float32), nf[keep].astype(np.int32), fc.astype(np.float32)


def import_crowd(D, cell=0.010):
    v, f, c = D["crowd_v"], D["crowd_f"], D["crowd_c"]
    if len(f) == 0:
        return
    if CROWD_COLOURS:
        # one colour per spectator (tools/blender/crowd_figures.py), the same crowd in every dump
        cc = np.load(CROWD_COLOURS)["crowd_c"]
        if len(cc) == len(c):
            c = cc
            print(f"[sumo_scene] crowd colours from {CROWD_COLOURS}")
        else:
            print(f"[sumo_scene] crowd colours ignored ({len(cc)} vertices, the dump has {len(c)})")
    n0 = len(f)
    if CROWD_CELL is not None:
        cell = CROWD_CELL
    if cell > 0:
        v, f, fc = _cluster(v, f, c, cell)
    else:
        # 10/20: the audience's meshes are simplified already (audience.py LOD_CELL, 4 mm near,
        # 10 mm far); clustering them again here broke the figures and mixed their clothes'
        # colours into a speckle (the user: the crowd looked in tatters). Kept as they are.
        v, f = v.astype(np.float32), f.astype(np.int32)
        fc = c[f[:, 0]].astype(np.float32)
    if LOOK >= 4:
        # a crowd on television is muted next to the ring: the colours half washed towards
        # their grey and a little darker (the figures' own were bright, a speckle)
        lum = fc @ np.array([0.30, 0.59, 0.11], np.float32)
        fc = ((lum[:, None] + 0.55 * (fc - lum[:, None])) * 0.85).astype(np.float32)
    print(f"[sumo_scene] crowd {n0} -> {len(f)} triangles ({cell * 1000:.0f} mm cells)")
    # look 5: flat shading -- the audience's simplified meshes have faces wound both ways, and
    # smooth normals over them shaded the shells in dark and light blotches (a camouflage)
    ob = _new_mesh_obj("crowd", v, f, None, smooth=LOOK < 5, validate=False)
    me = ob.data
    attr = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    lin = np.where(fc > 0.04045, fc ** 2.2, fc / 12.92).astype(np.float32)
    # the face's colour on its three corners, RGBA in float32 (float64 here was 1.1 GB)
    col = np.ones((len(lin) * 3, 4), np.float32)
    col[:, :3] = np.repeat(lin, 3, 0)
    del lin
    attr.data.foreach_set("color", col.ravel())
    del col
    if LOOK < 5:
        me.validate(clean_customdata=False)     # after the colours: it may drop faces, theirs with them
    # look 5: no validate -- dropping the degenerate faces shifted the corner colours after them
    # (the clothes' colours scattered over the figures)
    m = bpy.data.materials.new("crowd")
    m.use_nodes = True
    nt = m.node_tree
    a = nt.nodes.new("ShaderNodeVertexColor")
    a.layer_name = "Col"
    b = nt.nodes["Principled BSDF"]
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.7
    if "Sheen Weight" in b.inputs:
        b.inputs["Sheen Weight"].default_value = 0.4
    me.materials.append(m)
    if LOOK >= 5:
        # the crowd's own shadows on its faceted, overlapping pieces came out as blotches
        # (shadow acne over the stands' shells): it receives the hall's light, casts none
        for attr_name in ("visible_shadow",):
            if hasattr(ob, attr_name):
                setattr(ob, attr_name, False)
    print(f"[sumo_scene] crowd: {len(f)} triangles")



# ── the crowd, live (look 5) ───────────────────────────────────────────────────
# 10/20: the dump's crowd geom by geom (crowdg_*: blender_dump.py after 10/20) -- the 80
# piece meshes of the audience (audience.py's simplified Jumpers) mended once (their
# simplification left faces wound the wrong way: dark facets, a camouflage over the stands),
# the props (caps, towels, flags: boxes, cylinders, ellipsoids) too, and the whole crowd
# posed every frame as audience.py animated it (chatting, leaning in, cheering).

FRAMES = None          # the dump frame of each Blender frame (main sets it)


def _mend(v, f):
    """A piece mesh with its duplicate vertices merged and its faces wound outwards."""
    me = bpy.data.meshes.new("mend")
    me.from_pydata(v.tolist(), [], f.tolist())
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-4)
    # the holes the simplification tore (the shell's inside showed through them, dark)
    edges = [e for e in bm.edges if e.is_boundary]
    if edges:
        bmesh.ops.holes_fill(bm, edges=edges, sides=16)
        bmesh.ops.triangulate(bm, faces=bm.faces)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.verts.ensure_lookup_table()
    v2 = np.array([x.co[:] for x in bm.verts], np.float32)
    f2 = np.array([[x.index for x in fc.verts] for fc in bm.faces if len(fc.verts) == 3], np.int32)
    bm.free()
    bpy.data.meshes.remove(me)
    return v2, f2


def _unit(kind):
    """Unit primitives for the props, scaled per geom by its MuJoCo size."""
    if kind == 6:                                     # box: size = half extents
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=2.0)
    elif kind == 5:                                   # cylinder: radius, half height
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=True, segments=12, radius1=1.0, radius2=1.0, depth=2.0)
    else:                                             # ellipsoid / sphere: semi-axes
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=8, radius=1.0)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    v = np.array([x.co[:] for x in bm.verts], np.float32)
    f = np.array([[x.index for x in fc.verts] for fc in bm.faces], np.int32)
    bm.free()
    return v, f


def _qmat(q):
    """Quaternions (N, 4: w, x, y, z) to rotation matrices (N, 3, 3)."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], 1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], 1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)], 1)


class _LiveCrowd:
    def __init__(self, D):
        typ, mesh, size = D["crowdg_type"], D["crowdg_mesh"], D["crowdg_size"].astype(np.float32)
        rgba = D["crowdg_rgba"][:, :3].astype(np.float32).copy()
        # each spectator's pieces in audience.py's GROUPS order (belt = the chassis, top = the
        # shell, ...): the chassis, inside the shell, showed through the simplified shell in
        # dark spots -- it wears the shell's colour here (only a hint of it is ever seen)
        fig = D["crowdg_fig"]
        mesh_ids = np.where(typ == 7)[0]
        order = mesh_ids[np.lexsort((mesh_ids, fig[mesh_ids]))]
        start = np.r_[True, fig[order][1:] != fig[order][:-1]]
        first = order[start]                                    # each spectator's belt geom
        counts = np.diff(np.r_[np.where(start)[0], len(order)])
        ok = counts >= 2
        rgba[first[ok]] = rgba[order[np.where(start)[0][ok] + 1]]   # belt <- top
        self.pos = D["crowdg_pos"]
        self.quat = D["crowdg_quat"]
        self.every = int(D["crowdg_every"]) if "crowdg_every" in D else 1
        # templates: a mended mesh per piece mesh, a unit primitive per prop kind
        keys = [(int(t), int(m) if t == 7 else -1) for t, m in zip(typ, mesh)]
        tmpl = {}
        for k in sorted(set(keys)):
            tmpl[k] = _mend(D[f"mesh_{k[1]}_v"], D[f"mesh_{k[1]}_f"]) if k[0] == 7 else _unit(k[0])
        # normals: the simplified pieces' own facets shaded in dark blotches whatever was done
        # to them (folds, slivers, holes); a piece is shaded instead as the ellipsoid that fits
        # it (its covariance's), soft as cloth, the silhouette still the piece's own
        nrm = {}
        for k, (tv, tf) in tmpl.items():
            c = tv.mean(0)
            cov = np.cov((tv - c).T) + np.eye(3) * 1e-8
            g = (tv - c) @ np.linalg.inv(cov)
            nrm[k] = (g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)).astype(np.float32)
        # the soup's layout: geoms grouped by template, each group's vertices one block
        self.groups, verts, faces, cols, n = [], [], [], [], 0
        self.ngroups = []
        keys_arr = np.array([k[0] * 100000 + k[1] for k in keys])
        for k, (tv, tf) in tmpl.items():
            ids = np.where(keys_arr == k[0] * 100000 + k[1])[0]
            local = np.broadcast_to(tv, (len(ids),) + tv.shape).copy()
            if k[0] == 6 or k[0] == 4:
                local *= size[ids, None, :3]
            elif k[0] == 5:
                local *= np.stack([size[ids, 0], size[ids, 0], size[ids, 1]], 1)[:, None, :]
            self.groups.append((ids, local, n))
            self.ngroups.append((ids, np.broadcast_to(nrm[k], (len(ids),) + nrm[k].shape), n))
            nv = len(tv)
            faces.append((tf[None] + (n + nv * np.arange(len(ids)))[:, None, None]).reshape(-1, 3))
            cols.append(np.repeat(rgba[ids], len(tf), 0))
            n += nv * len(ids)
        self.nverts = n
        self.faces = np.concatenate(faces).astype(np.int32)
        self.fcol = np.concatenate(cols)
        print(f"[sumo_scene] live crowd: {len(typ)} geoms, {len(tmpl)} templates, {n} vertices, "
              f"{len(self.faces)} triangles, {len(self.pos)} poses (every {self.every} frames)")

    def normals(self, dump_frame: float) -> np.ndarray:
        k0 = int(np.clip(round(dump_frame / self.every), 0, len(self.pos) - 1))
        R = _qmat(self.quat[k0].astype(np.float32))
        out = np.empty((self.nverts, 3), np.float32)
        for ids, local, n in self.ngroups:
            w = np.einsum("gij,gvj->gvi", R[ids], local)
            out[n:n + w.size // 3] = w.reshape(-1, 3)
        return out

    def verts(self, dump_frame: float) -> np.ndarray:
        k = dump_frame / self.every
        k0 = int(np.clip(np.floor(k), 0, len(self.pos) - 1))
        k1 = min(k0 + 1, len(self.pos) - 1)
        a = float(np.clip(k - k0, 0.0, 1.0))
        p = (1 - a) * self.pos[k0] + a * self.pos[k1]
        q0, q1 = self.quat[k0].astype(np.float32), self.quat[k1].astype(np.float32)
        q1 = np.where((q0 * q1).sum(1, keepdims=True) < 0, -q1, q1)
        R = _qmat((1 - a) * q0 + a * q1)
        out = np.empty((self.nverts, 3), np.float32)
        for ids, local, n in self.groups:
            w = np.einsum("gij,gvj->gvi", R[ids], local) + p[ids, None, :]
            out[n:n + w.size // 3] = w.reshape(-1, 3)
        return out


def import_crowd_live(D):
    crowd = _LiveCrowd(D)
    fc = crowd.fcol
    if LOOK >= 4:
        lum = fc @ np.array([0.30, 0.59, 0.11], np.float32)
        fc = ((lum[:, None] + 0.55 * (fc - lum[:, None])) * 0.85).astype(np.float32)
    f0 = int(FRAMES[0]) if FRAMES is not None else 0
    ob = _new_mesh_obj("crowd", crowd.verts(f0), crowd.faces, None, smooth=True, validate=False)
    me = ob.data
    try:
        me.normals_split_custom_set_from_vertices(crowd.normals(f0))
    except Exception as e:                       # an older API: the mesh's own normals then
        print("[sumo_scene] crowd custom normals not set:", e)
    attr = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    lin = np.where(fc > 0.04045, fc ** 2.2, fc / 12.92).astype(np.float32)
    col = np.ones((len(lin) * 3, 4), np.float32)
    col[:, :3] = np.repeat(lin, 3, 0)
    attr.data.foreach_set("color", col.ravel())
    del col, lin
    m = bpy.data.materials.new("crowd")
    m.use_nodes = True
    nt = m.node_tree
    a = nt.nodes.new("ShaderNodeVertexColor")
    a.layer_name = "Col"
    b = nt.nodes["Principled BSDF"]
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.85        # cloth: no gloss on the facets
    me.materials.append(m)
    if hasattr(ob, "visible_shadow"):
        ob.visible_shadow = False                       # its own shadow darkened what holes remain
    # posed every frame from the dump (a frame handler: the render calls it per frame)
    frames = FRAMES

    def pose(scene, *_):
        i = min(max(scene.frame_current - 1, 0), len(frames) - 1) if frames is not None else 0
        me.vertices.foreach_set("co", crowd.verts(float(frames[i]) if frames is not None else 0.0).ravel())
        me.update()

    bpy.app.handlers.frame_change_pre.append(pose)
    return ob


def build_arena(M, D):
    import_static(D, M)
    if LOOK >= 5 and "crowdg_type" in D:
        import_crowd_live(D)
    else:
        import_crowd(D)
    build_dohyo(M)
    build_roof(M)
    if LOOK >= 5:
        detail_ring_props(M)
        detail_roof(M)
    floor = box("hall_floor", (0, 0, FLOOR - 0.025), (14, 14, 0.025), M["floor"])
    ceil = box("ceiling", (0, 0, 6.0), (14, 14, 0.05), M["dark_wood"])
    # banners hung from the ceiling round the hall: plain colours only, no names or marks
    rng = np.random.default_rng(3)
    for k in range(4):
        ang = k * math.pi / 2
        out = np.array([math.cos(ang), math.sin(ang)])
        along = np.array([-math.sin(ang), math.cos(ang)])
        for i in range(-5, 6):
            p = out * 5.6 + along * (i * 0.9)
            c = [(0.75, 0.12, 0.10), (0.10, 0.20, 0.55), (0.92, 0.90, 0.85), (0.15, 0.45, 0.25)][rng.integers(4)]
            box(f"kesho_banner_{k}_{i}", (p[0], p[1], 4.3), (0.006, 0.28, 0.9),
                _principled(f"banner_{c}", c, rough=0.8, sheen=0.5), rot=(0, 0, ang))
    lights_hall(M)
    if LOOK >= 5 and os.environ.get("JUMPER_BLENDER_HAZE", "1") == "1":
        hall_haze()


def lights_hall(M):
    """The ring lit hard from above the roof (the broadcast lights), the hall dimmer
    and warm, a faint blue ambience."""
    # the ring's lights hang inside the roof, as a hall's do, over the clay
    for i, (x, y) in enumerate(((0, 0), (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7))):
        L = bpy.data.lights.new(f"ring_light_{i}", "AREA")
        L.energy = (90 if i == 0 else 30) * (1.25 if LOOK >= 4 else 1.0)
        L.size = 0.6
        L.color = (1.0, 0.97, 0.92)
        ob = bpy.data.objects.new(f"ring_light_{i}", L)
        ob.location = (x, y, ROOF_EAVE - 0.12)
        bpy.context.scene.collection.objects.link(ob)
    for k in range(8):
        a = k * math.pi / 4
        L = bpy.data.lights.new(f"hall_light_{k}", "POINT")
        L.energy = 260 if LOOK >= 4 else 400       # look 4: the hall dimmer, the ring the bright place
        L.color = (1.0, 0.85, 0.65)
        L.shadow_soft_size = 0.6
        ob = bpy.data.objects.new(f"hall_light_{k}", L)
        ob.location = (5.0 * math.cos(a), 5.0 * math.sin(a), 4.5)
        bpy.context.scene.collection.objects.link(ob)
    world(0.02, (0.6, 0.65, 0.8))


def world(strength, colour=(1, 1, 1), sky=False):
    w = bpy.context.scene.world or bpy.data.worlds.new("world")
    bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    if sky:
        s = nt.nodes.new("ShaderNodeTexSky")
        try:
            s.sky_type = "NISHITA"
            s.sun_elevation = math.radians(35)
            s.sun_rotation = math.radians(130)
        except Exception:
            pass
        nt.links.new(s.outputs["Color"], bg.inputs["Color"])
    else:
        bg.inputs["Color"].default_value = (*colour, 1)
    bg.inputs["Strength"].default_value = strength


# ── the shrine ────────────────────────────────────────────────────────────────


def _cedar(name, x, y, h, M):
    cyl(f"{name}_trunk", (x, y, FLOOR + h * 0.25), 0.06 * h / 3, h * 0.5, M["bark"], verts=12)
    for i in range(5):
        r = (0.55 - 0.09 * i) * h / 3
        _add(bpy.ops.mesh.primitive_cone_add, f"{name}_crown_{i}", M["cedar"], False, vertices=10,
             radius1=r, radius2=0, depth=h * 0.32, location=(x, y, FLOOR + h * (0.38 + 0.13 * i)))


def _lantern(name, x, y, M):
    cyl(f"{name}_base", (x, y, FLOOR + 0.05), 0.09, 0.10, M["stone"], verts=6)
    cyl(f"{name}_post", (x, y, FLOOR + 0.30), 0.035, 0.40, M["stone"], verts=8)
    box(f"{name}_light", (x, y, FLOOR + 0.56), (0.07, 0.07, 0.06), M["stone"])
    box(f"{name}_window", (x, y, FLOOR + 0.56), (0.072, 0.04, 0.035), M["paper"])
    _add(bpy.ops.mesh.primitive_cone_add, f"{name}_cap", M["stone"], False, vertices=6, radius1=0.13,
         radius2=0.02, depth=0.09, location=(x, y, FLOOR + 0.665))


def build_shrine(M, D):
    build_dohyo(M)
    box("ground", (0, 0, FLOOR - 0.03), (30, 30, 0.03), M["gravel"])
    # the yakata: four posts and a thatched roof over the ring
    post = P_HALF + 0.12
    for sx in (-1, 1):
        for sy in (-1, 1):
            cyl(f"yakata_post_{sx}_{sy}", (sx * post, sy * post, FLOOR + 1.0), 0.045, 2.0 - FLOOR, M["wood"], verts=16)
    build_roof(M, eave=1.85, w=P_HALF * 1.08, hung=False)
    for ob in bpy.data.objects:
        if ob.name.startswith("roof_slope"):
            ob.data.materials[0] = M["thatch"]
    # shimenawa between the front posts, with paper shide
    sh = _principled("shimenawa", (0.80, 0.70, 0.45), rough=0.9)
    cyl("shimenawa", (0, -post, 1.55), 0.035, 2 * post, sh, rot=(0, math.pi / 2, 0), verts=12)
    for i in range(-2, 3):
        box(f"shide_{i}", (i * post * 0.4, -post - 0.01, 1.45), (0.03, 0.002, 0.07), M["white_cloth"])
    # the torii behind the muko side, cedars, lanterns, a lantern line
    ty = 5.5
    for sx in (-1, 1):
        cyl(f"torii_pillar_{sx}", (sx * 1.1, ty, FLOOR + 1.45), 0.09, 2.9, M["vermilion"], verts=24)
    box("torii_kasagi", (0, ty, FLOOR + 2.95), (1.75, 0.13, 0.08), M["black"], bevel=0.02)
    box("torii_shimaki", (0, ty, FLOOR + 2.83), (1.6, 0.11, 0.05), M["vermilion"])
    box("torii_nuki", (0, ty, FLOOR + 2.45), (1.45, 0.07, 0.06), M["vermilion"])
    # a cedar forest all round, clear of the approach behind the torii
    rng = np.random.default_rng(5)
    planted = 0
    while planted < 110:
        a = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(7.0, 19.0)
        x, y = d * math.cos(a), d * math.sin(a)
        if abs(x) < 3.2 and y > 3.0:
            continue
        _cedar(f"cedar_{planted}", x, y, rng.uniform(4.5, 8.5), M)
        planted += 1
    # the approach (sando): stone slabs from the torii to the hall
    for i in range(14):
        box(f"sando_{i}", (0, ty + 0.6 + i * 0.42, FLOOR + 0.005), (0.55, 0.19, 0.012), M["stone"], bevel=0.01)
    # the worship hall (haiden) at the approach's end: a platform, pillars, a copper roof
    hy = ty + 7.5
    green = _textured("copper_roof", (0.20, 0.42, 0.36), (0.30, 0.55, 0.47), 25, kind="wave", wave_dir="Y",
                      rough=0.55, bump=0.2)
    box("haiden_floor", (0, hy, FLOOR + 0.35), (2.6, 1.8, 0.35), M["wood"], bevel=0.02)
    for sx in (-1, -0.33, 0.33, 1):
        for sy in (-1, 1):
            cyl(f"haiden_pillar_{sx}_{sy}", (sx * 2.4, hy + sy * 1.6, FLOOR + 1.45), 0.09, 1.5, M["wood"], verts=16)
    box("haiden_wall", (0, hy + 1.6, FLOOR + 1.45), (2.4, 0.06, 0.75), M["plaster"])
    for s_ in (-1, 1):
        box(f"haiden_roof_{s_}", (0, hy - s_ * 1.25, FLOOR + 2.75), (3.2, 1.45, 0.08), green,
            rot=(s_ * math.radians(32), 0, 0), bevel=0.02)
    box("haiden_ridge", (0, hy, FLOOR + 3.55), (3.3, 0.14, 0.1), M["black"], bevel=0.02)
    _add(bpy.ops.mesh.primitive_uv_sphere_add, "suzu", M["gold"], True, radius=0.16,
         location=(0, hy - 1.75, FLOOR + 2.0))
    cyl("suzu_rope", (0, hy - 1.75, FLOOR + 1.3), 0.03, 1.3, _principled("suzu_rope", (0.85, 0.25, 0.2), rough=0.8))
    for i, (x, y) in enumerate(((-2.4, 3.2), (2.4, 3.2), (-2.4, -3.5), (2.4, -3.5))):
        _lantern(f"toro_{i}", x, y, M)
    # paper lanterns on a rope behind the ring (the muko side), seen behind the bout
    for sx in (-1, 1):
        cyl(f"chochin_pole_{sx}", (sx * 3.0, 3.6, FLOOR + 1.3), 0.04, 2.6 - FLOOR, M["wood"], verts=10)
    for i in range(-6, 7):
        _add(bpy.ops.mesh.primitive_uv_sphere_add, f"chochin_{i}", M["paper"], True, radius=0.09,
             location=(i * 0.45, 3.6, 2.25 - 0.18 * math.cos(i / 6 * math.pi / 2))).scale = (1, 1, 1.35)
    sun = bpy.data.lights.new("sun", "SUN")
    sun.energy = 5.0
    sun.color = (1.0, 0.93, 0.82)             # a warm afternoon sun
    sun.angle = math.radians(1.5)
    ob = bpy.data.objects.new("sun", sun)
    ob.rotation_euler = Euler((math.radians(50), 0, math.radians(140)))
    bpy.context.scene.collection.objects.link(ob)
    world(0.35, sky=True)          # less sky, more sun: contrast (10/20: the yard read pale)
    bpy.context.scene.view_settings.exposure = -2.2     # outdoors: the sky is bright


# ── the practice room ─────────────────────────────────────────────────────────


def build_keikoba(M, D):
    build_dohyo(M, sunk=False)
    room = 4.2
    box("keiko_floor", (0, 0, FLOOR - 0.02), (room, room, 0.02),
        _floorboards("floorboards", (0.46, 0.33, 0.21), (0.52, 0.38, 0.24)), apply=True)
    for k in range(4):
        ang = k * math.pi / 2
        out = np.array([math.cos(ang), math.sin(ang)])
        box(f"keiko_wall_{k}", (*(out * room), FLOOR + 1.3), (0.04, room, 1.3), M["plaster"], rot=(0, 0, ang))
        box(f"keiko_wainscot_{k}", (*(out * (room - 0.045)), FLOOR + 0.45), (0.01, room, 0.45), M["wood"],
            rot=(0, 0, ang))
        for i in range(-3, 4):
            p = out * (room - 0.05) + np.array([-math.sin(ang), math.cos(ang)]) * (i * 1.1)
            box(f"keiko_pillar_{k}_{i}", (p[0], p[1], FLOOR + 1.3), (0.05, 0.05, 1.3), M["dark_wood"], rot=(0, 0, ang))
    box("agari_zashiki", (0, room - 1.0, FLOOR + 0.18), (room, 0.95, 0.18), M["mat"], bevel=0.01)
    box("kamidana", (0, room - 0.12, FLOOR + 2.05), (0.6, 0.12, 0.03), M["wood"])
    box("kamidana_shrine", (0, room - 0.14, FLOOR + 2.2), (0.25, 0.09, 0.12), M["wood"], bevel=0.01)
    box("keiko_ceiling", (0, 0, FLOOR + 2.6), (room, room, 0.03), M["dark_wood"])
    for i in (-1, 0, 1):
        L = bpy.data.lights.new(f"window_{i}", "AREA")
        L.energy = 350
        L.size = 1.0
        L.color = (1.0, 0.95, 0.85)
        ob = bpy.data.objects.new(f"window_{i}", L)
        ob.location = (-room + 0.1, i * 1.4, FLOOR + 1.6)
        ob.rotation_euler = (0, math.radians(-75), 0)
        bpy.context.scene.collection.objects.link(ob)
    L = bpy.data.lights.new("keiko_lamp", "AREA")
    L.energy = 250
    L.size = 2.0
    ob = bpy.data.objects.new("keiko_lamp", L)
    ob.location = (0, 0, FLOOR + 2.5)
    bpy.context.scene.collection.objects.link(ob)
    world(0.05, (0.9, 0.85, 0.75))


# ── the robots ────────────────────────────────────────────────────────────────


def import_robots(D, frames):
    frames = np.asarray(frames)
    """Every moving geom an object, its world pose keyed per frame (F-curves set
    whole, not keyframe by keyframe)."""
    col = bpy.data.collections.new("robots")
    bpy.context.scene.collection.children.link(col)
    meshes: dict = {}
    pos, rot = D["dyn_pos"][frames], D["dyn_mat"][frames]
    F = len(frames)
    for g in range(len(D["dyn_type"])):
        if str(D["dyn_name"][g]).endswith(("_meshcol", "_col", "_collision")):
            continue                # collision copies of the visual meshes
        gt = int(D["dyn_type"][g])
        if gt == 7:
            mid = int(D["dyn_mesh"][g])
            if mid not in meshes:
                v, f = D[f"mesh_{mid}_v"], D[f"mesh_{mid}_f"]
                meshes[mid] = (v, f)
            v, f = meshes[mid]
        else:
            v, f = _prim_tris(gt, D["dyn_size"][g])
        rgba = D["dyn_rgba"][g]
        key = tuple(np.round(rgba[:3], 3))
        # shells and plastics: a little coat; dark parts rubbery
        bright = float(np.mean(rgba[:3]))
        mat = _principled(f"robot_{key}", rgba, rough=0.38 if bright > 0.2 else 0.6, coat=0.3 if bright > 0.2 else 0.0)
        ob = _new_mesh_obj(f"geom_{g}_{D['dyn_name'][g]}", v, f, mat, smooth=gt != 6, collection=col)
        if gt == 7:
            ob.data.set_sharp_from_angle(angle=math.radians(35)) if hasattr(ob.data, "set_sharp_from_angle") else None
        ob.rotation_mode = "QUATERNION"
        quats = np.array([Matrix(rot[i, g].reshape(3, 3).tolist()).to_quaternion()[:] for i in range(F)], np.float32)
        # keep the quaternions on one side so the interpolation does not flip
        for i in range(1, F):
            if np.dot(quats[i], quats[i - 1]) < 0:
                quats[i] = -quats[i]
        ob.animation_data_create()
        act = bpy.data.actions.new(f"act_{g}")
        ob.animation_data.action = act
        fr = np.arange(F, dtype=np.float32) + 1
        _set_curves(ob, act, "location", pos[:, g], fr)
        _set_curves(ob, act, "rotation_quaternion", quats, fr)
    print(f"[sumo_scene] robots: {len(D['dyn_type'])} geoms x {F} frames")


def _dressing(D, M):
    """A topknot (mage) on each wrestler's shell and a tall lacquered eboshi with a purple
    cord on the gyoji's, parented to the shell so they move with it (10/19)."""
    names = [str(n) for n in D["dyn_name"]]
    for ent in ("robot", "opponent", "gyoji"):
        try:
            gs = names.index(f"{ent}/upper_shell_link_visual")
            gd = names.index(f"{ent}/display_module_link_visual")
        except ValueError:
            continue
        shell = bpy.data.objects.get(f"geom_{gs}_{ent}/upper_shell_link_visual")
        if shell is None:
            continue
        R0 = D["dyn_mat"][0, gs].reshape(3, 3)
        up = R0.T @ np.array([0.0, 0.0, 1.0])
        v = D[f"mesh_{int(D['dyn_mesh'][gs])}_v"]
        top = v[np.argmax(v @ up)]
        centre = v.mean(0)
        face = R0.T @ (D["dyn_pos"][0, gd] - D["dyn_pos"][0, gs])     # toward the display, local
        fwd = face - up * (face @ up)
        fwd /= np.linalg.norm(fwd) + 1e-9
        side = np.cross(up, fwd)
        basis = Matrix(((fwd[0], side[0], up[0]), (fwd[1], side[1], up[1]), (fwd[2], side[2], up[2])))
        anchor = centre + up * ((top - centre) @ up)

        def child(name, op, mat, local, scale=(1, 1, 1), rot=(0, 0, 0), **kw):
            op(**kw)
            ob = bpy.context.active_object
            ob.name = f"{ent}_{name}"
            ob.data.materials.append(mat)
            bpy.ops.object.shade_smooth()
            ob.parent = shell
            ob.matrix_parent_inverse = Matrix.Identity(4)
            m = basis.to_4x4() @ Euler(rot).to_matrix().to_4x4()
            m.translation = Vector(anchor) + basis @ Vector(local)
            ob.matrix_basis = m @ Matrix.Diagonal((*scale, 1))
            return ob

        hair = _principled("bintsuke_hair", (0.02, 0.02, 0.025), rough=0.25, coat=0.8)
        if ent == "gyoji":
            child("eboshi", bpy.ops.mesh.primitive_cone_add, M["black"], (-0.01, 0, 0.040),
                  scale=(1.0, 0.75, 1.0), rot=(0, -0.25, 0), vertices=24, radius1=0.026, radius2=0.012, depth=0.08)
            cord = _principled("eboshi_cord", (0.45, 0.15, 0.55), rough=0.6, sheen=0.5)
            child("eboshi_cord", bpy.ops.mesh.primitive_torus_add, cord, (-0.004, 0, 0.006),
                  scale=(1.0, 0.8, 1.0), major_radius=0.026, minor_radius=0.003)
        else:
            child("mage_base", bpy.ops.mesh.primitive_uv_sphere_add, hair, (-0.02, 0, 0.004),
                  scale=(1.4, 1.0, 0.45), radius=0.016)
            child("mage", bpy.ops.mesh.primitive_cylinder_add, hair, (0.004, 0, 0.012),
                  rot=(0, math.pi / 2, 0), scale=(1.0, 0.9, 1.0), vertices=16, radius=0.007, depth=0.045)
            child("mage_tip", bpy.ops.mesh.primitive_uv_sphere_add, hair, (0.027, 0, 0.012), radius=0.008)
    print("[sumo_scene] dressing: mage x2, eboshi")


def _set_curves(ob, act, path, values, fr):
    n = values.shape[1]
    for i in range(n):
        try:
            fc = act.fcurves.new(path, index=i)
        except Exception:
            # Blender 5 layered actions: curves through the channelbag of a slot
            fc = _channelbag(ob, act).fcurves.new(path, index=i)
        fc.keyframe_points.add(len(fr))
        co = np.empty(len(fr) * 2, np.float32)
        co[0::2], co[1::2] = fr, values[:, i]
        fc.keyframe_points.foreach_set("co", co)
        fc.keyframe_points.foreach_set("interpolation", np.full(len(fr), 1, np.int32))   # LINEAR
        fc.update()


def _channelbag(ob, act):
    slot = ob.animation_data.action_slot or act.slots.new(id_type=ob.id_type, name=ob.name)
    ob.animation_data.action_slot = slot
    if not act.layers:
        layer = act.layers.new("layer")
        layer.strips.new(type="KEYFRAME")
    strip = act.layers[0].strips[0]
    cb = strip.channelbag(slot, ensure=True)
    return cb


# ── the camera ────────────────────────────────────────────────────────────────


def _smooth(k):
    return k * k * (3 - 2 * k)


# 10/20: the bout through a longer lens from further back (the replay's 45 degrees, its
# distance scaled to frame the same), as a broadcast camera far up the hall shows it: less
# wide-angle stretch, the crowd behind larger and soft (the focus follows the wrestlers).
TELE_FOVY = 30.0
TELE_FSTOP = 3.5


#: How far in the camera moves at the height of the claws' ceremony (a share of its distance).
CEREMONY_PUSH = 0.35


def ceremony_frames(D):
    """The dump frames of the claws' ceremony (upstream 9b019df), or None: from 0.8 s before
    the pincers' first shut to 0.7 s after their last, all before the first はっけよい (as
    sound_design.pincer_shuts finds them: a claw's tip closing on its palm after opening)."""
    caps = [str(c) for c in D["caption"]]
    names = [str(x) for x in D["dyn_name"]]
    hk = next((k for k, c in enumerate(caps) if c == "はっけよい"), len(caps))
    shuts = []
    for e in ("robot", "opponent"):
        for s in ("LF", "RF"):
            try:
                i = names.index(f"{e}/{s}_finger_tip_link_visual")
                j = names.index(f"{e}/{s}_palm_link_visual")
            except ValueError:
                continue
            d = np.linalg.norm(D["dyn_pos"][:hk, i] - D["dyn_pos"][:hk, j], axis=1)
            lo, hi = float(d.min()), float(d.max())
            if hi - lo < 0.01:
                continue
            opened = False
            for k, v in enumerate(d):
                if v > lo + 0.6 * (hi - lo):
                    opened = True
                elif opened and v < lo + 0.25 * (hi - lo):
                    shuts.append(k)
                    opened = False
    if not shuts:
        return None
    return max(0, min(shuts) - 24), max(shuts) + 21


def camera_from_dump(D, frames, lens_fovy=45.0, intro=0, wide=(0.0, 0.0, 0.25, 6.5, 125.0, -32.0), view="dump",
                     tele=TELE_FOVY, fstop=TELE_FSTOP):
    cam_d = bpy.data.cameras.new("cam")
    cam_d.sensor_fit = "VERTICAL"
    cam_d.sensor_height = 24.0
    cam_d.angle_y = math.radians(lens_fovy)
    # look 5: a tighter depth range (0.02..200 m left the audience's coincident parts fighting
    # in the depth buffer: blotches on the shells)
    cam_d.clip_start = 0.12 if LOOK >= 5 else 0.02
    cam_d.clip_end = 40 if LOOK >= 5 else 200
    cam_d.dof.use_dof = True
    cam_d.dof.aperture_fstop = fstop
    ob = bpy.data.objects.new("cam", cam_d)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    ob.rotation_mode = "QUATERNION"
    ob.animation_data_create()
    act = bpy.data.actions.new("cam_act")
    ob.animation_data.action = act
    C = D["cam"][frames].astype(np.float64).copy()
    if view != "dump":
        # another angle (the replay's): the two wrestlers' midpoint, smoothed, seen low
        # from the shomen side ("low") or from the muko side ("muko", the reverse angle)
        names = [str(n) for n in D["dyn_name"]]
        ids = [names.index(f"{e}/base_link_visual") for e in ("robot", "opponent")]
        mid = D["dyn_pos"][frames][:, ids].mean(1).astype(np.float64)
        sm = mid.copy()
        for i in range(1, len(sm)):
            sm[i] = sm[i - 1] + 0.08 * (mid[i] - sm[i - 1])
        az, el, dist, z = {"low": (90.0, -7.0, 0.85, 0.09), "muko": (-90.0, -16.0, 1.2, 0.08)}[view]
        C = np.c_[sm[:, 0], sm[:, 1], np.full(len(sm), z), np.full(len(sm), dist), np.full(len(sm), az),
                  np.full(len(sm), el)]
    # the intro: frames that hold the first pose while the camera cranes down from
    # a wide view of the hall (az 125 -> the replay's, distance 6.5 -> its own), the lens
    # going from wide to long on the way
    fovs = np.full(len(C), float(tele))
    # the claws' ceremony (a replay with JUMPER_SUMO_CEREMONY, 10/21): the camera moves in on
    # the pair while the pincers clap and the claws go up in a V, and back for the shikiri
    win = ceremony_frames(D) if view == "dump" else None
    if win is not None:
        k0, k1 = win
        for i in range(intro, len(C)):
            f = frames[i]
            if k0 <= f <= k1:
                C[i, 3] *= 1.0 - CEREMONY_PUSH * math.sin(math.pi * (f - k0) / max(1, k1 - k0)) ** 2
    for i in range(intro):
        k = _smooth(i / max(1, intro - 1))
        C[i] = np.asarray(wide) + k * (C[intro] - np.asarray(wide))
        fovs[i] = lens_fovy + k * (tele - lens_fovy)
    locs, quats, focus = [], [], []
    for (lx, ly, lz, dist, az, el), fov in zip(C, fovs):
        dist = dist * math.tan(math.radians(lens_fovy) / 2) / math.tan(math.radians(fov) / 2)
        a, e = math.radians(az), math.radians(el)
        fwd = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        look = Vector((lx, ly, lz))
        p = look - fwd * dist
        q = fwd.to_track_quat("-Z", "Y")
        locs.append(p[:])
        quats.append(q[:])
        focus.append(dist)
    quats = np.array(quats, np.float32)
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    fr = np.arange(len(C), dtype=np.float32) + 1
    _set_curves(ob, act, "location", np.array(locs, np.float32), fr)
    _set_curves(ob, act, "rotation_quaternion", quats, fr)
    # the lens and the focus, frame by frame (the focus on the look-at point, the wrestlers)
    cam_d.animation_data_create()
    lact = bpy.data.actions.new("lens_act")
    cam_d.animation_data.action = lact
    lens = cam_d.sensor_height / 2 / np.tan(np.radians(fovs) / 2)
    _set_curves(cam_d, lact, "lens", lens.astype(np.float32)[:, None], fr)
    _set_curves(cam_d, lact, "dof.focus_distance", np.array(focus, np.float32)[:, None], fr)
    return ob


# ── render settings ───────────────────────────────────────────────────────────


def setup_render(engine, samples, res, out):
    sc = bpy.context.scene
    w, h = (int(x) for x in res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = out
    sc.render.fps = 30
    # a render going on after a crash (render_queue2.sh's retries): the frames written are kept
    sc.render.use_overwrite = os.environ.get("JUMPER_BLENDER_RESUME", "0") != "1"
    items = [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    if engine in ("cycles", "cycles_cpu"):
        sc.render.engine = "CYCLES"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for dev in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = dev
                prefs.get_devices()
                for d in prefs.devices:
                    d.use = d.type == dev
                break
            except Exception:
                continue
        sc.cycles.device = "CPU" if engine == "cycles_cpu" else "GPU"
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 6
    else:
        sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in items else "BLENDER_EEVEE"
        ee = sc.eevee
        rt = os.environ.get("JUMPER_BLENDER_RT", "1") == "1"
        for attr, val in (("taa_render_samples", samples), ("use_raytracing", rt), ("use_shadows", True),
                          ("use_gtao", rt), ("use_bloom", True), ("shadow_pool_size", "1024")):
            if hasattr(ee, attr):
                setattr(ee, attr, val)
    sc.view_settings.view_transform = "AgX" if "AgX" in [v.identifier for v in
                                                          sc.view_settings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
    for look in ("AgX - Punchy", "AgX - Medium High Contrast"):
        try:
            sc.view_settings.look = look
            break
        except Exception:
            continue


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--venue", default="arena", choices=("arena", "shrine", "keikoba"))
    ap.add_argument("--out", default="//frames/")
    ap.add_argument("--frames", default="")
    ap.add_argument("--still", type=int, default=-1)
    ap.add_argument("--engine", default="eevee")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--res", default="1920x1080")
    ap.add_argument("--save", default="")
    ap.add_argument("--intro", type=float, default=0.0, help="seconds held still while the camera closes in")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--exposure", type=float, default=None, help="override the venue's exposure")
    ap.add_argument("--camera", default="dump", choices=("dump", "low", "muko"),
                    help="dump: the replay's own camera; low / muko: another angle (for the replay)")
    ap.add_argument("--look", type=int, default=int(os.environ.get("JUMPER_BLENDER_LOOK", "3")),
                    help="3: as M0094; 4: crowd muted, hall dimmer (default from JUMPER_BLENDER_LOOK)")
    ap.add_argument("--crowd-colours", default=os.environ.get("JUMPER_BLENDER_CROWD", ""),
                    help="crowd_figures.py's colours, one per spectator (default from JUMPER_BLENDER_CROWD)")
    ap.add_argument("--crowd-cell", type=float, default=None,
                    help="the crowd's clustering cell in m (default 0.010; 0 with look 5: the meshes as they are)")
    ap.add_argument("--portraits", default=os.environ.get("JUMPER_BLENDER_PORTRAITS", ""),
                    help="portraits.py's pictures (default from JUMPER_BLENDER_PORTRAITS)")
    a = ap.parse_args(argv)
    global LOOK, CROWD_COLOURS, CROWD_CELL, PORTRAITS
    PORTRAITS = a.portraits
    LOOK = a.look
    CROWD_COLOURS = a.crowd_colours
    CROWD_CELL = a.crowd_cell if a.crowd_cell is not None else (0.0 if LOOK >= 5 else None)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    D = dict(np.load(a.dump, allow_pickle=False))
    n = len(D["t"])
    if a.frames:
        s, e = (int(x) for x in a.frames.split(":"))
        frames = np.arange(max(0, s), min(n, e))
    else:
        frames = np.arange(n)
    intro = int(round(a.intro * 30))
    frames = np.concatenate([np.full(intro, frames[0]), frames])
    global FRAMES
    FRAMES = frames
    M = mats()
    {"arena": build_arena, "shrine": build_shrine, "keikoba": build_keikoba}[a.venue](M, D)
    import_robots(D, frames)
    _dressing(D, M)
    # the intro's wide view per venue: the practice room's ceiling is 2.6 m up
    wide = {"arena": (0.0, 0.0, 0.25, 6.5, 125.0, -32.0), "shrine": (0.0, 0.0, 0.25, 6.5, 125.0, -26.0),
            "keikoba": (0.0, 0.0, 0.15, 3.0, 120.0, -20.0)}[a.venue]
    camera_from_dump(D, frames, intro=intro, view=a.camera, wide=wide)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, len(frames)
    setup_render(a.engine, a.samples, a.res, a.out)
    if a.exposure is not None:
        sc.view_settings.exposure = a.exposure
    elif a.engine == "eevee":
        # EEVEE, after test stills (10/19 23:45: the user found the first videos too white)
        sc.view_settings.exposure = EXPOSURE_EEVEE[a.venue]
    if a.save:
        bpy.ops.wm.save_as_mainfile(filepath=a.save)
    if a.no_render:
        return
    if a.still >= 0:
        sc.frame_set(a.still + 1)
        sc.render.filepath = a.out if a.out.endswith(".png") else a.out + f"still_{a.still:05d}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[sumo_scene] still -> {sc.render.filepath}")
    else:
        bpy.ops.render.render(animation=True)
        print(f"[sumo_scene] {len(frames)} frames -> {a.out}")


if __name__ == "__main__":
    main()
