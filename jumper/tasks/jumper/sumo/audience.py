"""The hall round the dohyo, for replays (jumper-challenge-2026): judges, spectators
in their seats, and the roof over the ring.

Everyone is a Jumper built from the robot's own visual meshes (simplified for a
crowd), seated, dressed in colours of its own, and moving: the judges sit with
their claws folded and watch the bout; the spectators sit on cushions and chairs,
chat during the shikiri, lean in as the bout goes on and cheer when it is decided
-- each in its own way and at its own moment. Decoration only (geom group 3: no
mass, no contact, out of every sensor's sight).

- **Shimpan.** Five judges in black on cushions at the foot of the mound: the
  chief judge on the shomen side, two on the mukō-jōmen side (where the gyoji
  stands), one on each of east and west (docs/notes/dohyo-and-kimarite.md).
- **Seats.** Two rows of tamari-seki cushions on the floor, four rows of masu-seki
  boxes (four to a box, behind pipe rails) on rising steps, and chairs on a
  balcony above; aisles at the corners.
- **The roof.** A shinmei-style suspended roof over the ring with the purple
  mizuhiki-maku round its eaves and the four coloured tassels at its corners.

The motion: a replay's renderer calls `Crowd` (installed as `env.sumo_decorate`
by the gyoji) once per frame, after the physics; it poses every figure by forward
kinematics of a seated Jumper and writes the geoms' world poses into the render
data. Nothing it does reaches the simulation.

Turned on by `JUMPER_SUMO_AUDIENCE=1` in a replay (`env_cfg.py`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

ROBOT_XML = Path(__file__).resolve().parents[3] / "assets" / "jumper" / "jumper.xml"

#: Seated: the base frame this high above the seat, its belly (0.9 cm below the
#: frame) a little above the cushion, the four walking legs folded with the knees up.
SIT_Z = 0.024
SIT_LEGS = (1.2, -2.4)            # J1, J2 of the left legs; the right ones mirrored
#: Arm poses (left arm J0..J4; the right arm mirrors J0, J2, J3, J4), found by
#: searching the claw tip's position over the arm's joints (10/10 night).
ARMS = {
    "lap": (0.30, -1.50, -0.68, -2.10, 0.0),     # claws resting low in front
    "front": (-0.85, -2.30, -0.30, -1.64, 0.0),  # claws up before the chest
    "clap": (-1.15, -2.30, -0.30, -1.64, 0.0),   # ... and together
    "face": (-1.20, -2.40, -0.30, -1.64, 0.0),   # claws over the face
    "up": (-0.30, -3.20, -1.82, 0.21, 0.0),      # raised high (banzai)
    "wave": (-0.15, -3.20, -1.44, 0.21, 0.0),    # raised and out to the side
    "fold": (-1.20, -1.95, -0.30, -1.90, 0.0),   # folded before the body (judges)
}
#: Crowd meshes: the robot's visual meshes simplified by vertex clustering (cell, m).
#: The full robot is 431 000 triangles; a hall of crabs needs far fewer each.
LOD_CELL = {"near": 0.004, "far": 0.010}
KID_SCALE = 0.8

# Colours. A spectator's parts become clothes: the red shell a jacket or kimono,
# the red arm parts sleeves, the red leg parts trousers, the silver forearms and
# thighs a shirt or skin, the white pads shoes and gloves, the black belly a belt.
# No red shells: those are the wrestlers'.
TOPS = [(0.12, 0.16, 0.30), (0.22, 0.22, 0.24), (0.07, 0.07, 0.08), (0.78, 0.70, 0.56),
        (0.90, 0.90, 0.88), (0.55, 0.70, 0.85), (0.42, 0.45, 0.25), (0.42, 0.28, 0.18),
        (0.55, 0.55, 0.57), (0.88, 0.62, 0.68), (0.66, 0.58, 0.80), (0.85, 0.68, 0.22),
        (0.18, 0.48, 0.50), (0.45, 0.12, 0.20), (0.20, 0.22, 0.45), (0.95, 0.80, 0.84),
        (0.30, 0.50, 0.38), (0.90, 0.50, 0.18), (0.35, 0.25, 0.40), (0.60, 0.75, 0.55)]
BOTTOMS = [(0.10, 0.12, 0.22), (0.08, 0.08, 0.09), (0.35, 0.35, 0.37), (0.62, 0.56, 0.40),
           (0.25, 0.35, 0.55), (0.35, 0.24, 0.16), (0.20, 0.20, 0.22), (0.85, 0.82, 0.72),
           (0.36, 0.38, 0.22), (0.45, 0.45, 0.48)]
LIGHTS = [(0.86, 0.80, 0.72), (0.93, 0.93, 0.91), (0.72, 0.62, 0.52), (0.62, 0.66, 0.72),
          (0.82, 0.75, 0.62), (0.95, 0.88, 0.80), (0.55, 0.45, 0.38)]
SHOES = [(0.06, 0.06, 0.06), (0.30, 0.18, 0.10), (0.92, 0.92, 0.90), (0.45, 0.45, 0.47),
         (0.62, 0.48, 0.30)]
JUDGE = {"top": (0.05, 0.05, 0.06), "arms": (0.05, 0.05, 0.06), "legs": (0.32, 0.32, 0.35),
         "light": (0.92, 0.92, 0.90), "shoe": (0.95, 0.95, 0.95), "belt": (0.03, 0.03, 0.03)}
#: Towel colours for the two sides' fans: the wrestlers' mawashi (env_cfg.MAWASHI_RGBA).
SIDE_RGBA = ((0.13, 0.27, 0.72), (0.10, 0.52, 0.30))
CUSHION = (0.42, 0.16, 0.50)       # judges' zabuton
TAMARI = (0.50, 0.12, 0.14)
MASU_CUSHION = (0.66, 0.24, 0.12)
WOOD, DARK_WOOD = (0.42, 0.30, 0.20), (0.20, 0.14, 0.10)
MAT = (0.66, 0.62, 0.44)
RAIL = (0.78, 0.78, 0.80)
CHAIR = (0.55, 0.10, 0.12)
ROOF, GOLD, MAKU = (0.30, 0.24, 0.18), (0.80, 0.64, 0.26), (0.36, 0.14, 0.48)
#: The four tassels, by corner (x sign, y sign). Shomen is -y, east is -x:
#: blue at shomen-east, black at shomen-west, red at muko-east, white at muko-west.
TASSELS = {(-1, -1): (0.12, 0.40, 0.42), (1, -1): (0.06, 0.06, 0.07),
           (-1, 1): (0.78, 0.12, 0.10), (1, 1): (0.93, 0.93, 0.90)}



# ── The seated Jumper, simplified ──────────────────────────────────────────


def _decimate(v: np.ndarray, f: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: every vertex in a `cell`-sized cube merged into one."""
    key = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = int(inv.max()) + 1
    v2 = np.zeros((n, 3))
    np.add.at(v2, inv, v)
    v2 /= np.bincount(inv, minlength=n)[:, None]
    f2 = inv[f]
    f2 = f2[(f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])]
    if len(f2):
        _, first = np.unique(np.sort(f2, axis=1), axis=0, return_index=True)
        f2 = f2[np.sort(first)]
    used = np.unique(f2)
    remap = -np.ones(n, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v2[used].astype(np.float32), remap[f2].astype(np.int32)


@lru_cache(maxsize=1)
def _robot() -> tuple[mujoco.MjModel, tuple[int, ...]]:
    """The robot as compiled, and its visual mesh geoms."""
    m = mujoco.MjSpec.from_file(str(ROBOT_XML)).compile()
    parts = tuple(g for g in range(m.ngeom)
                  if m.geom_group[g] == 2 and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    return m, parts


def _groups() -> list[tuple[str, str, list[str]]]:
    """A crowd Jumper in 20 pieces instead of 41: parts that never move against
    each other and wear one colour are one mesh. (colour, the body the piece rides
    on, the bodies whose visual meshes it is made of.) Seated, the walking legs
    never move against the body."""
    legs = ("LM", "RM", "LR", "RR")
    groups = [
        ("belt", "base_link", ["base_link"]),
        ("top", "base_link", ["upper_shell_link"]),
        ("face", "base_link", ["display_module_link", "tof_sensor_link", "camera_link"]),
        ("legs", "base_link", [f"{l}_{p}_link" for l in legs for p in ("hip", "calf")]),
        ("light", "base_link", [f"{l}_thigh_link" for l in legs]),
        ("shoe", "base_link", [f"{l}_foot_tip_link" for l in legs]),
    ]
    for s in ("LF", "RF"):
        groups += [
            ("arms", f"{s}_shoulder_link", [f"{s}_shoulder_link"]),
            ("arms", f"{s}_upper_arm_link", [f"{s}_upper_arm_link"]),
            ("light", f"{s}_forearm_link", [f"{s}_forearm_link"]),
            ("arms", f"{s}_palm_link", [f"{s}_palm_link"]),
            ("light", f"{s}_palm_link", [f"{s}_palm_pad_f_link", f"{s}_palm_pad_b_link",
                                         f"{s}_palm_grip_insert_link"]),
            ("arms", f"{s}_finger_link", [f"{s}_finger_link"]),
            ("light", f"{s}_finger_link", [f"{s}_finger_tip_link", f"{s}_finger_grip_insert_link"]),
        ]
    return groups


GROUPS = _groups()
#: Bodies props hang from (`Figure.props`' body index): the shell, the two palms.
PROP_BODIES = ("upper_shell_link", "LF_palm_link", "RF_palm_link")


def _mirror(arm: tuple) -> tuple:
    j0, j1, j2, j3, j4 = arm
    return (-j0, j1, -j2, -j3, -j4)


class _Kinematics:
    """Forward kinematics of a seated Jumper: arm angles in; out, the pose of each
    piece's body (`GROUPS`) and of the prop bodies, in the figure's frame. Cached
    by the arm angles, rounded."""

    def __init__(self):
        from ..common.constants import HOME

        self.m, _ = _robot()
        self.d = mujoco.MjData(self.m)
        jid = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, n)
        adr = lambda n: self.m.jnt_qposadr[jid(n)]
        self.arm_adr = np.array([adr(f"{s}F_J{k}_joint") for s in "LR" for k in range(5)])
        q = self.d.qpos
        q[:] = 0.0
        q[2], q[3] = SIT_Z, 1.0
        for j, v in HOME.items():
            q[adr(j)] = v
        for leg, sign in (("LM", 1), ("LR", 1), ("RM", -1), ("RR", -1)):
            q[adr(f"{leg}_J1_joint")] = sign * SIT_LEGS[0]
            q[adr(f"{leg}_J2_joint")] = sign * SIT_LEGS[1]
        q[adr("LM_J0_joint")] = q[adr("RM_J0_joint")] = 0.0
        body = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.bodies = np.array([body(b) for _, b, _ in GROUPS] + [body(b) for b in PROP_BODIES])
        self.cache: dict = {}

    def __call__(self, arms: np.ndarray):
        key = tuple(np.round(arms / 0.025).astype(int))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        self.d.qpos[self.arm_adr] = arms
        mujoco.mj_kinematics(self.m, self.d)
        out = (self.d.xpos[self.bodies].copy(), self.d.xmat[self.bodies].reshape(-1, 3, 3).copy())
        if len(self.cache) > 8000:
            self.cache.clear()
        self.cache[key] = out
        return out


@lru_cache(maxsize=1)
def _kin() -> _Kinematics:
    return _Kinematics()


def _mesh(spec, name: str, v: np.ndarray, f: np.ndarray) -> None:
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = v.reshape(-1).tolist()
    mesh.userface = f.reshape(-1).tolist()
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL


@lru_cache(maxsize=None)
def _lod(cell: float) -> dict:
    """The 20 pieces at one level of detail: each piece's simplified, merged mesh in
    its body's frame, and where MuJoCo puts that mesh's geom frame when it compiles
    it (`frame_pos`, `frame_mat`, in the body's frame).

    MuJoCo moves a mesh geom's frame to the mesh's centre of mass and principal
    axes; a pose for a crowd geom is its body's pose composed with that offset.
    (Placed as if the frame stayed where it was given, the first crowd came out in
    pieces, 10/10.)
    """
    m0, parts = _robot()
    kin = _kin()
    kin(np.array([*ARMS["lap"], *_mirror(ARMS["lap"])]))
    d = kin.d
    by_body = {mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_BODY, int(m0.geom_bodyid[p])): p for p in parts}
    verts, faces = [], []
    for _, ref, members in GROUPS:
        b = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, ref)
        vs, fs, n = [], [], 0
        for member in members:
            p = by_body[member]
            mid = int(m0.geom_dataid[p])
            va, vn = m0.mesh_vertadr[mid], m0.mesh_vertnum[mid]
            fa, fn = m0.mesh_faceadr[mid], m0.mesh_facenum[mid]
            c = cell
            while True:
                v, f = _decimate(m0.mesh_vert[va:va + vn], m0.mesh_face[fa:fa + fn], c)
                if len(v) >= 8 and len(f) >= 8:
                    break
                c /= 2
            world = d.geom_xpos[p] + v @ d.geom_xmat[p].reshape(3, 3).T
            vs.append((world - d.xpos[b]) @ d.xmat[b].reshape(3, 3))
            fs.append(f + n)
            n += len(v)
        verts.append(np.concatenate(vs).astype(np.float32))
        faces.append(np.concatenate(fs).astype(np.int32))
    # Where each merged mesh's geom frame lands.
    spec = mujoco.MjSpec()
    holder = spec.worldbody.add_body(name="holder")
    for i, (v, f) in enumerate(zip(verts, faces)):
        _mesh(spec, f"piece_{i}", v, f)
        holder.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"piece_{i}", contype=0,
                        conaffinity=0, mass=0.0)
    m = spec.compile()
    frame_mat = np.zeros((len(GROUPS), 9))
    for i in range(len(GROUPS)):
        mujoco.mju_quat2Mat(frame_mat[i], m.geom_quat[i])
    return dict(verts=verts, faces=faces, frame_pos=m.geom_pos[:len(GROUPS)].copy(),
                frame_mat=frame_mat.reshape(-1, 3, 3), faces_total=int(sum(len(f) for f in faces)))


# ── Figures ───────────────────────────────────────────────────────────────


@dataclass
class Figure:
    tag: str
    kind: str                  # "judge" or "fan"
    lod: str
    scale: float
    pos: np.ndarray            # seat point, world
    yaw: float                 # facing the ring
    colours: dict
    props: list = field(default_factory=list)   # (body index, pos, mat, geom kwargs)
    # Temperament (fans): when, how and how much it reacts.
    threshold: float = 0.5     # excitement it takes to start cheering
    side: int = -1             # 0 east's fan, 1 west's, -1 neither
    style: str = "clap"        # its way of cheering
    hand: int = 0              # 0 left, 1 right: the arm that waves or pumps
    tempo: float = 2.0         # Hz
    phase: float = 0.0
    rate: float = 3.0          # 1/s: how quickly it takes up the hall's mood
    look: float = 0.7          # how far it turns to follow the bout
    chat_period: float = 9.0
    chat_phase: float = 0.0
    chat_dir: float = 1.0
    lap: tuple = (0.0, 0.0)    # its own resting arm offsets (J0, J1)


#: Filled by `add_audience` while the scene is built; read by `Crowd`.
_FIGURES: list[Figure] = []


def _rgba(c) -> list[float]:
    return [*c[:3], 1.0]


def _piece_colour(key: str, colours: dict) -> list[float]:
    """A piece's colour: the face stays the robot's black, the rest are clothes."""
    return _rgba((0.02, 0.02, 0.02) if key == "face" else colours[key])


def _quat(mat: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.reshape(-1))
    return q.tolist()


def _rot(yaw: float, pitch: float = 0.0) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    return rz @ ry


def _rest_arms(fig: Figure) -> np.ndarray:
    if fig.kind == "judge":
        left = ARMS["fold"]
    else:
        l = ARMS["lap"]
        left = (l[0] + fig.lap[0], l[1] + fig.lap[1], *l[2:])
    return np.array([*left, *_mirror(left)])


def _place(spec, fig: Figure, meshes: dict) -> None:
    """The figure's geoms in its seated rest pose (what a viewer without the
    renderer's hook shows), in a body `decor_crowd_<tag>`: the 20 pieces in `GROUPS`
    order, then the props. The geoms are unnamed and few: MjSpec's cost of adding
    an element grows with the elements already there, and a first hall of 45 000
    named geoms took five minutes to build (10/10 night)."""
    from .env_cfg import DECOR_GROUP

    vis = dict(contype=0, conaffinity=0, mass=0.0, group=DECOR_GROUP)
    xb, rb = _kin()(_rest_arms(fig))
    body = _hall(spec).add_body(name=f"decor_crowd_{fig.tag}", pos=fig.pos.tolist(),
                                   quat=[math.cos(fig.yaw / 2), 0, 0, math.sin(fig.yaw / 2)])
    names = meshes[(fig.lod, fig.scale)]
    for i, (key, _, _) in enumerate(GROUPS):
        # Given at its body's pose: compiling adds the mesh's own frame offset.
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=names[i],
                      pos=(fig.scale * xb[i]).tolist(), quat=_quat(rb[i]),
                      rgba=_piece_colour(key, fig.colours), **vis)
    n = len(GROUPS)
    for k, (b, lp, lm, kw) in enumerate(fig.props):
        p = xb[n + b] + rb[n + b] @ lp
        body.add_geom(pos=(fig.scale * p).tolist(), quat=_quat(rb[n + b] @ lm),
                      **{**kw, "size": [fig.scale * v for v in kw["size"]]}, **vis)


def _fan(rng: random.Random, tag: str, lod: str, pos, yaw: float, kid_ok: bool = True) -> Figure:
    top = rng.choice(TOPS)
    suit = rng.random() < 0.35
    colours = {"top": top, "arms": top if suit or rng.random() < 0.5 else rng.choice(BOTTOMS),
               "legs": top if suit else rng.choice(BOTTOMS), "light": rng.choice(LIGHTS),
               "shoe": rng.choice(SHOES), "belt": rng.choice(BOTTOMS[:3] + [top])}
    kid = kid_ok and rng.random() < 0.10
    side = rng.choices((0, 1, -1), weights=(0.35, 0.35, 0.30))[0]
    fig = Figure(
        tag=tag, kind="fan", lod=lod, scale=KID_SCALE if kid else 1.0,
        pos=np.array(pos, dtype=float), yaw=yaw, colours=colours,
        threshold=rng.uniform(0.12, 0.80), side=side,
        style=rng.choices(("clap", "banzai", "wave", "pump", "snap"),
                          weights=(0.32, 0.20, 0.18, 0.16, 0.14))[0],
        hand=rng.randrange(2), tempo=rng.uniform(1.6, 3.0), phase=rng.uniform(0, 2 * math.pi),
        rate=rng.uniform(1.5, 6.0), look=rng.uniform(0.45, 0.9),
        chat_period=rng.uniform(6.0, 14.0), chat_phase=rng.uniform(0, 2 * math.pi),
        chat_dir=rng.choice((-1.0, 1.0)), lap=(rng.uniform(-0.2, 0.2), rng.uniform(-0.25, 0.2)))
    # Props: a cap or a hat on some; a fan's towel in the claw of some.
    if rng.random() < 0.22:
        c = _rgba(rng.choice(TOPS + BOTTOMS))
        top_local = np.array([0.0535, 0.0, 0.0376])          # shell top, shell frame
        fig.props.append((0, top_local + [0, 0, 0.004], np.eye(3),
                          dict(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.034, 0.034, 0.016], rgba=c)))
        brim = 0.045 if rng.random() < 0.4 else 0.0
        if brim:   # a hat with a brim all round
            fig.props.append((0, top_local, np.eye(3),
                              dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[brim, 0.0015, 0], rgba=c)))
        else:      # a cap's peak, forward
            fig.props.append((0, top_local + [0.035, 0, -0.001], np.eye(3),
                              dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.018, 0.022, 0.0015], rgba=c)))
    if side >= 0 and rng.random() < 0.35:
        # A towel held up in the waving claw, along it: palm frame +y is the claw.
        fig.props.append((1 + fig.hand, np.array([0.0, 0.19, -0.012]), np.eye(3),
                          dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.028, 0.055, 0.0015],
                               rgba=_rgba(SIDE_RGBA[side]))))
    return fig


def _judge(tag: str, pos, yaw: float) -> Figure:
    return Figure(tag=tag, kind="judge", lod="near", scale=1.0, pos=np.array(pos, dtype=float),
                  yaw=yaw, colours=dict(JUDGE), look=0.8)


# ── The hall ──────────────────────────────────────────────────────────────


def _hall(spec):
    """The body every static part of the hall hangs in, `decor_hall`. Named
    `decor_*`, as the figures are, so the native backend can keep the whole hall
    out of the physics model and draw it all the same (mjrl native_sim
    `_split_decor`): 23 000 geoms in the physics model cost every step and every
    ray of the height scan, and the first hall rendered at 5 steps a second."""
    hit = _HALLS.get(id(spec))
    if hit is None or hit[0] is not spec:
        hit = _HALLS[id(spec)] = (spec, spec.worldbody.add_body(name="decor_hall"))
    return hit[1]


_HALLS: dict = {}


def _box(spec, name, pos, half, rgba, yaw=0.0, **extra) -> None:
    """A box of the hall. `name` is only for reading the code: the hall's few
    thousand boxes are left unnamed, as the crowd's geoms are (`_place`)."""
    from .env_cfg import DECOR_GROUP

    del name
    _hall(spec).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half),
                            pos=list(pos), quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)],
                            rgba=_rgba(rgba), contype=0, conaffinity=0, mass=0.0,
                            group=DECOR_GROUP, **extra)


def _side(k: int):
    """Side k (0 west +x, 1 muko +y, 2 east -x, 3 shomen -y): outward unit vector,
    along unit vector, and the yaw that faces the ring from it."""
    a = k * math.pi / 2
    out = np.array([math.cos(a), math.sin(a)])
    along = np.array([-math.sin(a), math.cos(a)])
    return out, along, a + math.pi


def _seats(spec, floor_z: float, mound_half: float, rng: random.Random) -> list[Figure]:
    figs: list[Figure] = []
    # Tamari-seki: two rows of cushions on the floor behind the judges.
    for row, dist in enumerate((mound_half + 0.62, mound_half + 0.95)):
        for k in range(4):
            out, along, yaw = _side(k)
            span = dist - 0.30
            n = int(2 * span // 0.34)
            for i in range(n):
                t = -span + (i + 0.5) * 2 * span / n
                xy = out * dist + along * t
                _box(spec, f"tamari_{row}_{k}_{i}", [*xy, floor_z + 0.008], (0.13, 0.13, 0.008), TAMARI, yaw)
                figs.append(_fan(rng, f"t{row}{k}_{i}", "near", [*xy, floor_z + 0.016],
                                 yaw + rng.uniform(-0.08, 0.08)))
    # Masu-seki: boxes of four on rising steps, pipe rails between them.
    box, rows, rise = 0.72, 4, 0.07
    r0 = mound_half + 1.30
    for row in range(rows):
        inner = r0 + row * box
        top = floor_z + rise * (row + 1)
        for k in range(4):
            out, along, yaw = _side(k)
            span = inner - 0.25
            n = int(2 * span // box)
            centre = out * (inner + box / 2)
            # The step under this row, the whole side long.
            _box(spec, f"masu_step_{row}_{k}", [*centre, (floor_z + top) / 2],
                 (box / 2, span + 0.05, (top - floor_z) / 2), WOOD, yaw)
            for i in range(n):
                t = -span + (i + 0.5) * 2 * span / n
                c = centre + along * t
                _box(spec, f"masu_mat_{row}_{k}_{i}", [*c, top + 0.002], (box / 2 - 0.01, box / 2 - 0.01, 0.002), MAT, yaw)
                for e, (du, dv) in enumerate(((1, 0), (0, 1))):   # two rails per box
                    p = c + out * (du * box / 2) + along * (dv * box / 2)
                    half = (0.004, box / 2, 0.003) if du else (box / 2, 0.004, 0.003)
                    _box(spec, f"masu_rail_{row}_{k}_{i}_{e}", [*p, top + 0.03], half, RAIL, yaw)
                for j, (u, v) in enumerate(((-0.17, -0.17), (-0.17, 0.17), (0.17, -0.17), (0.17, 0.17))):
                    p = c + out * u + along * v
                    _box(spec, f"masu_zabuton_{row}_{k}_{i}_{j}", [*p, top + 0.009], (0.12, 0.12, 0.007),
                         MASU_CUSHION, yaw)
                    figs.append(_fan(rng, f"m{row}{k}_{i}_{j}", "far", [*p, top + 0.016],
                                     yaw + rng.uniform(-0.12, 0.12)))
    # The balcony: a front wall, then three rows of chairs rising behind it.
    rb = r0 + rows * box + 0.20
    wall_top = floor_z + 0.80
    for k in range(4):
        out, along, yaw = _side(k)
        _box(spec, f"balcony_front_{k}", [*(out * rb), (floor_z + wall_top) / 2],
             (0.04, rb + 0.05, (wall_top - floor_z) / 2), DARK_WOOD, yaw)
        _box(spec, f"balcony_band_{k}", [*(out * (rb - 0.045)), wall_top - 0.06],
             (0.004, rb, 0.035), MAKU, yaw)
        for row in range(3):
            d = rb + 0.30 + 0.40 * row
            seat = wall_top - 0.05 + 0.14 * row
            _box(spec, f"balcony_step_{k}_{row}", [*(out * d), (floor_z + seat) / 2 - 0.02],
                 (0.20, d - 0.2, (seat - floor_z) / 2 - 0.02), DARK_WOOD, yaw)
            span = d - 0.45
            n = int(2 * span // 0.36)
            for i in range(n):
                t = -span + (i + 0.5) * 2 * span / n
                p = out * d + along * t
                _box(spec, f"chair_{k}_{row}_{i}", [*p, seat - 0.01], (0.13, 0.13, 0.012), CHAIR, yaw)
                back = out * (d + 0.14) + along * t
                _box(spec, f"chair_back_{k}_{row}_{i}", [*back, seat + 0.07], (0.012, 0.13, 0.08), CHAIR, yaw)
                figs.append(_fan(rng, f"b{k}{row}_{i}", "far", [*p, seat + 0.002],
                                 yaw + rng.uniform(-0.1, 0.1), kid_ok=False))
        # The hall's wall behind, and the champions' portraits along it.
        dw = rb + 1.65
        _box(spec, f"hall_wall_{k}", [*(out * dw), floor_z + 1.6], (0.05, dw + 0.05, 1.6), (0.16, 0.11, 0.08), yaw)
        for i in range(-4, 5):
            p = out * (dw - 0.06) + along * (i * 0.85)
            _box(spec, f"portrait_frame_{k}_{i}", [*p, floor_z + 2.35], (0.01, 0.26, 0.34), GOLD, yaw)
            _box(spec, f"portrait_{k}_{i}", [*(p - out * 0.012), floor_z + 2.35], (0.004, 0.22, 0.30),
                 (0.30 + 0.05 * (i % 3), 0.24, 0.20), yaw)
    return figs


def _roof(spec, mound_half: float) -> None:
    """A shinmei-style roof hung over the ring: gabled, a ridge beam with katsuogi
    logs across it and chigi crossed at the gables, the purple mizuhiki-maku round
    the eaves and a tassel at each corner. Its height is not in the sources found
    (10/10); it is hung clear of the bout camera's view, about 5 m up, scaled."""
    from .env_cfg import DECOR_GROUP

    vis = dict(contype=0, conaffinity=0, mass=0.0, group=DECOR_GROUP)
    w = mound_half * 1.02            # eaves, half width
    eave = 1.80                      # eave height above the clay
    pitch = math.radians(28)
    run = w * 1.08
    rise = run * math.tan(pitch)
    for s in (-1, 1):                # the two slopes, ridge along x
        q = [math.cos(s * pitch / 2), math.sin(s * pitch / 2), 0, 0]
        _hall(spec).add_geom(name=f"roof_slope_{s}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[w * 1.12, run / math.cos(pitch) / 2, 0.025],
                                pos=[0, -s * run / 2, eave + rise / 2], quat=q, rgba=_rgba(ROOF), **vis)
    ridge = eave + rise + 0.02
    _hall(spec).add_geom(name="roof_ridge", type=mujoco.mjtGeom.mjGEOM_BOX, size=[w * 1.14, 0.05, 0.04],
                            pos=[0, 0, ridge], rgba=_rgba(ROOF), **vis)
    for i in range(7):               # katsuogi
        x = -w * 0.9 + i * (1.8 * w / 6)
        _hall(spec).add_geom(name=f"katsuogi_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.035, 0.09, 0],
                                pos=[x, 0, ridge + 0.07], quat=[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0],
                                rgba=_rgba(GOLD), **vis)
    for sx in (-1, 1):               # chigi: two boards crossed above each gable
        for s in (-1, 1):
            a = s * math.radians(35)
            _hall(spec).add_geom(name=f"chigi_{sx}_{s}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[0.012, 0.03, 0.22], pos=[sx * w * 1.12, s * 0.07, ridge + 0.08],
                                    quat=[math.cos(a / 2), math.sin(a / 2), 0, 0], rgba=_rgba(ROOF), **vis)
    # Mizuhiki-maku: the purple band round the eaves, with a gilt edge above.
    for k in range(4):
        out, along, yaw = _side(k)
        _hall(spec).add_geom(name=f"mizuhiki_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.006, w, 0.10], pos=[*(out * w), eave - 0.08],
                                quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], rgba=_rgba(MAKU), **vis)
        _hall(spec).add_geom(name=f"eave_trim_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=[0.012, w + 0.01, 0.02], pos=[*(out * w), eave + 0.03],
                                quat=[math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], rgba=_rgba(GOLD), **vis)
    # The four tassels.
    for (sx, sy), c in TASSELS.items():
        x, y = sx * w, sy * w
        _hall(spec).add_geom(name=f"fusa_knot_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.045, 0, 0],
                                pos=[x, y, eave - 0.16], rgba=_rgba(c), **vis)
        _hall(spec).add_geom(name=f"fusa_cord_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.018, 0.07, 0], pos=[x, y, eave - 0.07], rgba=_rgba(c), **vis)
        _hall(spec).add_geom(name=f"fusa_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                size=[0.055, 0.15, 0], pos=[x, y, eave - 0.34], rgba=_rgba(c), **vis)
    # Light: the ring lit from under the roof, the hall more dimly from above.
    _hall(spec).add_light(name="ring_light", pos=[0, 0, eave - 0.05], dir=[0, 0, -1],
                             diffuse=[0.22, 0.21, 0.19], specular=[0.05, 0.05, 0.05], cutoff=55,
                             exponent=2, castshadow=True)
    for i, (x, y) in enumerate(((3.5, 3.5), (-3.5, -3.5))):
        _hall(spec).add_light(name=f"hall_light_{i}", pos=[x, y, 3.2], dir=[-x, -y, -3.2],
                                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.28, 0.27, 0.25],
                                 specular=[0.0, 0.0, 0.0], castshadow=False)
    # The cables it hangs from.
    for sx in (-1, 1):
        for sy in (-1, 1):
            _hall(spec).add_geom(name=f"roof_cable_{sx}_{sy}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                                    size=[0.006, 1.2, 0], pos=[sx * w * 0.6, sy * 0.25, ridge + 1.2],
                                    rgba=[0.3, 0.3, 0.3, 1.0], **vis)


def add_audience(spec, floor_z: float, mound_half: float, seed: int = 10) -> None:
    """The judges, the seats and their spectators, and the roof, round a mound
    `mound_half` m square whose foot is at `floor_z`."""
    rng = random.Random(seed)
    figs: list[Figure] = []
    # The judges, on cushions at the foot of the mound: shomen (-y) the chief judge,
    # muko-jomen (+y) two, east (-x) and west (+x) one each.
    d = mound_half + 0.30
    for i, (x, y) in enumerate(((0.0, -d), (-0.55, d), (0.55, d), (-d, 0.0), (d, 0.0))):
        yaw = math.atan2(-y, -x)
        _box(spec, f"zabuton_{i}", [x, y, floor_z + 0.01], (0.17, 0.17, 0.01), CUSHION, yaw)
        figs.append(_judge(f"judge{i}", [x, y, floor_z + 0.02], yaw))
    figs += _seats(spec, floor_z, mound_half, rng)
    _roof(spec, mound_half)
    # The crowd's meshes: per LOD, and a child's size where children sit.
    meshes: dict = {}
    for lod, cell in LOD_CELL.items():
        data = _lod(cell)
        for scale in sorted({f.scale for f in figs if f.lod == lod}):
            names = []
            for i, (v, f) in enumerate(zip(data["verts"], data["faces"])):
                name = f"crowd_{lod}_{int(scale * 100)}_{i}"
                _mesh(spec, name, v * scale, f)
                names.append(name)
            meshes[(lod, scale)] = names
    for fig in figs:
        _place(spec, fig, meshes)
    _FIGURES[:] = figs


# ── The crowd, moving ─────────────────────────────────────────────────────


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


class _BatchKinematics:
    """`_Kinematics` for every figure at once, in numpy: the seated body is fixed,
    so only the two arm chains move (J0 about z, J1 about y, J2..J4 about x, each
    at its body's origin), and a chain of matrix products does it. Output bodies in
    `GROUPS` then `PROP_BODIES` order, like `_Kinematics`."""

    def __init__(self):
        kin = _kin()
        m, d = kin.m, kin.d
        kin(np.array([*ARMS["lap"], *_mirror(ARMS["lap"])]))      # any arms: the body is fixed
        body = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.chain = {}
        for side in "LR":
            names = [f"{side}F_{p}_link" for p in ("shoulder", "upper_arm", "forearm", "palm", "finger")]
            ids = [body(n) for n in names]
            offs, rots, axes = [], [], []
            for b in ids:
                offs.append(m.body_pos[b].copy())
                r = np.zeros(9)
                mujoco.mju_quat2Mat(r, m.body_quat[b])
                rots.append(r.reshape(3, 3))
                axes.append(m.jnt_axis[m.body_jntadr[b]].copy())
            parent = m.body_parentid[ids[0]]
            self.chain[side] = dict(names=names, offs=offs, rots=rots, axes=axes,
                                    root_pos=d.xpos[parent].copy(),
                                    root_mat=d.xmat[parent].reshape(3, 3).copy())
        self.order = [b for _, b, _ in GROUPS] + list(PROP_BODIES)
        self.static = {n: (d.xpos[body(n)].copy(), d.xmat[body(n)].reshape(3, 3).copy())
                       for n in self.order}

    @staticmethod
    def _axis_rot(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotations about one unit axis by angles q [F] -> [F, 3, 3]."""
        x, y, z = axis
        k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
        return np.eye(3) + s * k + (1 - c) * (k @ k)

    def __call__(self, arms: np.ndarray):
        """arms [F, 10] -> (pos [F, B, 3], mat [F, B, 3, 3]) for `self.order`."""
        f = arms.shape[0]
        poses = {}
        for s, side in enumerate("LR"):
            c = self.chain[side]
            p = np.broadcast_to(c["root_pos"], (f, 3))
            r = np.broadcast_to(c["root_mat"], (f, 3, 3))
            for k, name in enumerate(c["names"]):
                p = p + r @ c["offs"][k]
                r = r @ c["rots"][k] @ self._axis_rot(c["axes"][k], arms[:, 5 * s + k])
                poses[name] = (p, r)
        pos = np.empty((f, len(self.order), 3))
        mat = np.empty((f, len(self.order), 3, 3))
        for i, n in enumerate(self.order):
            if n in poses:
                pos[:, i], mat[:, i] = poses[n]
            else:
                pos[:, i], mat[:, i] = self.static[n]
        return pos, mat


def _mirror_rows(left: np.ndarray) -> np.ndarray:
    """Left-arm angles [F, 5] -> both arms [F, 10], the right mirrored."""
    return np.concatenate((left, left * np.array([-1.0, 1.0, -1.0, -1.0, -1.0])), -1)


class Crowd:
    """`env.sumo_decorate`: called by the renderer with its model and data after each
    frame's physics; poses every figure for this moment of the bout -- all of them
    at once, in numpy (one figure at a time took 76 ms a frame for a thousand)."""

    STYLE = {"clap": 0, "banzai": 1, "wave": 2, "pump": 3, "snap": 4}

    def __init__(self, env):
        self.env = env
        figs = list(_FIGURES)
        self.figs = figs
        n = len(figs)
        self.n = n
        a = lambda k, dt=float: np.array([getattr(f, k) for f in figs], dtype=dt)
        self.pos = np.array([f.pos for f in figs], dtype=float).reshape(n, 3)
        self.yaw0 = a("yaw")
        self.scale = a("scale")
        self.judge = np.array([f.kind == "judge" for f in figs], dtype=bool)
        self.style = np.array([self.STYLE.get(f.style, 0) for f in figs], dtype=int)
        self.hand = a("hand", int)
        self.tempo, self.phase, self.rate = a("tempo"), a("phase"), a("rate")
        self.look, self.threshold, self.side = a("look"), a("threshold"), a("side", int)
        self.chat_period, self.chat_phase, self.chat_dir = a("chat_period"), a("chat_phase"), a("chat_dir")
        self.rest = np.array([_rest_arms(f) for f in figs], dtype=float).reshape(n, 10)
        lod = np.array([0 if f.lod == "near" else 1 for f in figs], dtype=int)
        self.mood = np.zeros(n)
        self.turn = np.zeros(n)
        self.last_t = None
        self.ids = None
        self.ok = n > 0
        if not n:
            return
        self.kin = _BatchKinematics()
        lods = [_lod(LOD_CELL["near"]), _lod(LOD_CELL["far"])]
        self.frame_pos = np.stack([l["frame_pos"] for l in lods])[lod]     # [F, G, 3]
        self.frame_mat = np.stack([l["frame_mat"] for l in lods])[lod]     # [F, G, 3, 3]
        # Props: figure, body (after the pieces' bodies), local pos and mat.
        props = [(i, b, lp, lm) for i, f in enumerate(figs) for b, lp, lm, _ in f.props]
        g = len(GROUPS)
        self.prop_fig = np.array([p[0] for p in props], dtype=int)
        self.prop_body = np.array([g + p[1] for p in props], dtype=int)
        self.prop_pos = np.array([p[2] for p in props], dtype=float).reshape(-1, 3)
        self.prop_mat = np.array([p[3] for p in props], dtype=float).reshape(-1, 3, 3)
        self.pose = {k: np.array([*v, *_mirror(v)], dtype=float) for k, v in ARMS.items()}

    def _resolve(self, model) -> None:
        pieces, props = [], []
        for f in self.figs:
            n = len(GROUPS) + len(f.props)
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"decor_crowd_{f.tag}")
            if b < 0 or model.body_geomnum[b] != n:
                self.ok = False
                print("[audience] crowd geoms not found in the render model: not animated")
                return
            adr = int(model.body_geomadr[b])
            pieces.append(adr + np.arange(len(GROUPS)))
            props += [adr + len(GROUPS) + k for k in range(len(f.props))]
        self.ids = np.array(pieces).reshape(-1)
        self.prop_ids = np.array(props, dtype=int)

    def _bout(self) -> dict:
        """The moment of the bout in environment 0, as the crowd sees it."""
        env = self.env
        from .mdp import bout

        r = getattr(env, "sumo_referee", None)
        st = bout(env)
        origin = env.scene.env_origins[0, :2]
        a = (env.scene["robot"].data.root_link_pos_w[0, :2] - origin).cpu().numpy()
        b = (env.scene["opponent"].data.root_link_pos_w[0, :2] - origin).cpu().numpy()
        out = dict(started=bool(st.started[0]), decided=False, since=0.0, winner=-1,
                   mid=(a + b) / 2, edge=float(max(np.linalg.norm(a), np.linalg.norm(b))),
                   confirmed=False)
        if r is not None:
            out["decided"] = bool(r.decided[0])
            out["since"] = float(r.time[0] - r.decided_at[0])
            out["winner"] = int(r.winner[0])
            out["confirmed"] = bool(r._env0.get("confirmed", False))
        return out

    def __call__(self, model, data) -> None:
        if self.ok and self.ids is None:
            self._resolve(model)
        if not self.ok:
            return
        env = self.env
        # The crowd's clock: the renderer's own while it holds the world still for
        # its opening move (`sumo_clock`), then the world's, carried on from there.
        clock = getattr(env, "sumo_clock", None)
        if clock is not None:
            t = float(clock)
        else:
            if getattr(self, "_counter0", None) is None:
                self._counter0 = int(env.common_step_counter)
            t = (int(env.common_step_counter) - self._counter0) * float(env.step_dt) \
                + float(getattr(env, "sumo_clock_offset", 0.0))
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), 0.2)
        self.last_t = t
        s = self._bout()
        # The hall's mood: quiet through the shikiri, rising at "matta nashi",
        # up through the bout (more near the straw), loudest at the decision.
        if s["decided"]:
            target = 1.0 if s["since"] < 1.4 else 0.6
        elif s["started"]:
            target = 0.55 + 0.4 * float(_smooth((s["edge"] - 0.40) / 0.35))
        else:
            target = 0.35 if s["confirmed"] else 0.10
        self.mood += (target - self.mood) * (1 - np.exp(-self.rate * dt))
        arms, look, lean, lift = self._motion(t, s)
        # Turn to follow the bout, smoothly.
        want = np.arctan2(s["mid"][1] - self.pos[:, 1], s["mid"][0] - self.pos[:, 0]) - self.yaw0
        want = (want + np.pi) % (2 * np.pi) - np.pi
        want = self.look * np.clip(want, -0.6, 0.6) + look
        if dt:
            self.turn += (want - self.turn) * (1 - math.exp(-4.0 * dt))
        bp, bm = self.kin(arms)                                     # [F, B, 3], [F, B, 3, 3]
        g = len(GROUPS)
        local = bp[:, :g] + np.einsum("fgij,fgj->fgi", bm[:, :g], self.frame_pos)
        mats = bm[:, :g] @ self.frame_mat
        yaw = self.yaw0 + self.turn
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(lean), np.sin(lean)
        rot = np.zeros((self.n, 3, 3))                              # Rz(yaw) @ Ry(lean)
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 0, 2] = cy * cp, -sy, cy * sp
        rot[:, 1, 0], rot[:, 1, 1], rot[:, 1, 2] = sy * cp, cy, sy * sp
        rot[:, 2, 0], rot[:, 2, 2] = -sp, cp
        offset = self.pos.copy()
        offset[:, 2] += lift
        pos = np.einsum("fij,fgj->fgi", rot, local * self.scale[:, None, None]) + offset[:, None]
        data.geom_xpos[self.ids] = pos.reshape(-1, 3)
        data.geom_xmat[self.ids] = (rot[:, None] @ mats).reshape(-1, 9)
        if len(self.prop_ids):
            fi, bi = self.prop_fig, self.prop_body
            ppos = bp[fi, bi] + np.einsum("pij,pj->pi", bm[fi, bi], self.prop_pos)
            ppos = np.einsum("pij,pj->pi", rot[fi], ppos * self.scale[fi, None]) + offset[fi]
            data.geom_xpos[self.prop_ids] = ppos
            data.geom_xmat[self.prop_ids] = (rot[fi] @ bm[fi, bi] @ self.prop_mat).reshape(-1, 9)

    def _motion(self, t: float, s: dict):
        """(arm angles [F, 10], extra head turn, lean, lift, each [F]) for everyone now."""
        n = self.n
        P = self.pose
        rest = self.rest
        breathe = 0.012 * np.sin(0.9 * t + self.phase)
        look = 0.04 * np.sin(0.31 * t + self.phase)
        lean = breathe.copy()
        lift = np.zeros(n)
        w = 2 * np.pi * self.tempo * t + self.phase
        osc = np.sin(w)
        pulse = 0.5 + 0.5 * osc
        a = _smooth((self.mood - self.threshold) / 0.25)
        target = np.broadcast_to(P["front"], (n, 10)).copy()
        style = self.style.copy()
        if not s["decided"]:
            style[style == 1] = 3              # the banzai is kept for the decision
        front_l = np.broadcast_to(P["front"][:5], (n, 5))
        # clap: the claws swung together and apart
        m = style == 0
        clap_l = front_l.copy()
        clap_l[:, 0] = P["front"][0] + (P["clap"][0] - P["front"][0]) * pulse
        target[m] = _mirror_rows(clap_l)[m]
        # banzai: both claws up, bobbing, snapping
        m = style == 1
        up_l = np.broadcast_to(P["up"][:5], (n, 5)).copy()
        up_l[:, 1] += 0.18 * osc
        up_l[:, 4] = -0.9 * pulse
        target[m] = _mirror_rows(up_l)[m]
        lift[m] = 0.010 * pulse[m]
        # wave: one claw up and waving, the other before the chest
        m = style == 2
        wave_l = np.broadcast_to(P["wave"][:5], (n, 5)).copy()
        wave_l[:, 0] += 0.35 * osc
        waved = _mirror_rows(wave_l)
        left, right = m & (self.hand == 0), m & (self.hand == 1)
        target[left, :5] = waved[left, :5]
        target[right, 5:] = waved[right, 5:]
        # pump: one claw driven up and down, the other resting
        m = style == 3
        pumped = _mirror_rows(front_l + pulse[:, None] * (P["up"][:5] - P["front"][:5]))
        left, right = m & (self.hand == 0), m & (self.hand == 1)
        target[left, :5], target[left, 5:] = pumped[left, :5], rest[left, 5:]
        target[right, 5:], target[right, :5] = pumped[right, 5:], rest[right, :5]
        lift[m] = 0.006 * pulse[m]
        # snap: claws before the chest, opening and closing
        m = style == 4
        snap_l = front_l.copy()
        snap_l[:, 4] = -1.1 * pulse
        target[m] = _mirror_rows(snap_l)[m]
        arms = rest + a[:, None] * (target - rest)
        lift = lift * a
        if s["started"] and not s["decided"]:
            lean = lean + 0.12 * a             # leaning in to the bout
        if not s["started"]:
            # Calm: now and then one turns to its neighbour and talks with a claw.
            c = _smooth((np.sin(2 * np.pi * t / self.chat_period + self.chat_phase) - 0.55) / 0.3)
            c = c * (a < 0.05)
            look = look + self.chat_dir * 0.55 * c
            talk = 0.5 * P["front"] + 0.5 * rest
            talk[:, 4] = -0.5 * pulse
            talk[:, 9] = 0.5 * pulse
            left = self.hand == 0
            arms[left, :5] += c[left, None] * (talk[left, :5] - arms[left, :5])
            arms[~left, 5:] += c[~left, None] * (talk[~left, 5:] - arms[~left, 5:])
        if s["decided"] and s["winner"] >= 0 and s["since"] < 2.2:
            # Their wrestler lost: claws to the face, slumped.
            k = float(_smooth(s["since"] / 0.3))
            m = self.side == 1 - s["winner"]
            arms[m] += k * (P["face"] - arms[m])
            lean[m] += 0.20 * k
            lift[m] = 0.0
        # The judges: claws folded, still, a nod at the decision.
        j = self.judge
        arms[j] = rest[j]
        lift[j] = 0.0
        lean[j] = breathe[j] + (0.10 * math.sin(math.pi * s["since"] / 0.6)
                                if (s["decided"] and s["since"] < 0.6) else 0.0)
        look[j] = 0.0
        return arms, look, lean, lift
