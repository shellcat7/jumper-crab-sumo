"""jumper.sumo: two Jumpers in a ring (jumper-challenge-2026).

Built on `jumper.posture` so the learner is exactly the posture robot -- the same
415 observations and 20 actions -- and a posture checkpoint loads into it as is.
What this file adds:

- **A second robot**, `opponent`, a copy of the same entity facing the first across
  the ring, driven by a frozen posture policy (`opponent.py`). It walks at the
  learner; it is not trained.
- **Collisions between the two.** The posture masks give each leg its own bit so
  that different legs of one robot collide and the same leg never does
  (`common/constants.py`). With two copies that scheme lets bodies pass through
  each other entirely: base vs base, base vs legs and same-named legs all come out
  0. Each robot therefore gets one more bit pair, its own in `contype` and the
  other's in `conaffinity`, which makes every geom of one collide with every geom
  of the other while leaving each robot's own pairs exactly as before.
- **A ring**: visual only (no contact), centred on the origin. Every environment's
  origin is put on the origin (`env_spacing = 0`): environments are separate
  worlds, so they never touch, and one ring then serves all of them.
- **Episode ends** when either robot leaves the ring or either falls over.
- **Training** (`play=False`): the velocity command becomes "walk at the opponent"
  and the posture command neutral, and the bout's outcome is paid on top of the
  posture rewards (`mdp.py`). The robot's physics (mass, joints, motors) are not
  changed.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os

import mujoco
from pathlib import Path


from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..posture.env_cfg import env_cfg as posture_env_cfg
from .arms import RitualJointPositionActionCfg
from .gyoji import GYOJI, POSTURE_GYOJI, TWIST_GYOJI, GyojiPolicyActionCfg, _dress
from .mdp import (
    ApproachCommandCfg, bout_outcome, down_on_clay, out_of_ring, push_out, reset_bout, straw_edge,
)
#: A wrestler is down when one of these touches the clay: the body and its
#: fittings, and the upper segments of the arms and legs.
DOWN_BODIES = ("base_link", "upper_shell_link", "display_module_link", "tof_sensor_link",
               "camera_link", ".*_shoulder_link", ".*_upper_arm_link", ".*_forearm_link",
               ".*_hip_link", ".*_thigh_link")
from .referee import bout_over, referee_watch, reset_referee
from .opponent import (
    OPPONENT, POSTURE_OPP, TWIST_OPP, HeldCommandCfg, OpponentPolicyActionCfg,
    opponent_observation_group,
)

#: Ring radius, metres. The robot is 0.33 m long; a real dohyo is about nine
#: wrestlers across, and this is about five robots.
RING_RADIUS = 0.8
#: Each robot starts this far from the centre, facing the other. Crouched for the
#: tachiai a Jumper's claw tips reach about 0.19 m ahead of its base (measured
#: 10/10), and in a real bout the hands go down behind the shikiri line, whose
#: inner edge is _LINE_GAP / 2 = 0.1225 m out. In the shikiri crouch a wrestler
#: also settles a few cm forward of where it was put (measured: west from 0.34
#: to 0.32, its claws at 0.111-0.125), so 0.36 (+-0.02 of start noise) keeps the
#: claws behind the line. (0.25 until 10/10 put them 3 cm over it.)
START_X = 0.36
#: The frozen policy driving the opponent. The official posture example by default.
OPPONENT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "posture" / "out" / "example" / "model_74800.pt"
)
#: Collision channel bits: above the ground bit and the six leg bits.
_BIT_ROBOT = 1 << 8
_BIT_OPPONENT = 1 << 9
_BIT_GYOJI = 1 << 10
#: Standing height, as in posture.
_NEUTRAL_HEIGHT = 0.10647
#: What a bout's end is worth, in the units of the posture rewards (whose total
#: is about 135 over a 20 s episode): winning about four seconds of good walking.
WIN_REWARD = 30.0
LOSS_REWARD = 30.0
#: Weight on the opponent's outward speed (m/s, clamped to 0.5).
PUSH_OUT_WEIGHT = 4.0


def _with_channel(collision, own: int, other: int):
    c = copy.deepcopy(collision)
    c.contype = {k: v | own for k, v in c.contype.items()}
    c.conaffinity = {k: v | other for k, v in c.conaffinity.items()}
    return c


#: A real dohyo scaled to this ring (circle 4.55 m across -> 1.6 m, k = 0.35): a
#: wrestler about 0.85 m across becomes this 0.3 m robot. Every part below uses the
#: same k. Sources and the figures they disagree on: docs/notes/dohyo-and-kimarite.md.
DOHYO_SCALE = RING_RADIUS / 2.275
_PLATFORM_HALF = 6.7 / 2 * DOHYO_SCALE        # clay mound, 6.7 m square
_PLATFORM_HEIGHT = 0.45 * DOHYO_SCALE         # 34-66 cm in the sources; 45 cm taken
_BALE_RADIUS = 0.07 * DOHYO_SCALE             # straw bale about 14 cm thick ...
_BALE_SHOWING = 0.04 * DOHYO_SCALE            # ... of which 4 cm shows above the clay
_SAND_WIDTH = 0.25 * DOHYO_SCALE              # janome: sand outside the circle
_MOUND_RUN = 0.30 * DOHYO_SCALE               # the sides' slope, out over the height
_LINE_LENGTH, _LINE_WIDTH, _LINE_GAP = (0.90 * DOHYO_SCALE, 0.06 * DOHYO_SCALE,
                                       0.70 * DOHYO_SCALE)
#: Decorations live in geom group 3: renderers hide it unless asked (render_offscreen
#: turns it on), and ray-cast sensors (foot height scan, ToF) only read groups 0-2,
#: so nothing here can reach an observation or a reward.
DECOR_GROUP = 3


#: Mawashi colours: east (the robot, the learner) blue, west (the opponent) green,
#: so the three red Jumpers in a replay can be told apart.
MAWASHI_RGBA = {"robot": (0.13, 0.27, 0.72, 1.0), OPPONENT: (0.10, 0.52, 0.30, 1.0)}


def recolor(spec, geom: str, rgba) -> None:
    """Give a geom of the robot's own a colour (`entity/name`): its parts recoloured
    rather than parts added, as a skin does. The official skins (jumper-design's
    `.skin`) are for the Web simulator; this training checkout has no importer for
    them (its WORKFLOWS.md), so a replay recolours the MJCF's visual geoms."""
    for g in spec.geoms:
        if g.name == geom:
            g.rgba = list(rgba)
            g.material = ""


def _mawashi(spec) -> None:
    """Each wrestler's belly -- the black base under the red shell, where a mawashi
    sits -- in its side's colour. (First drawn as an added slab round the body; the
    user asked for the robot's own parts recoloured instead, 10/10.)"""
    for entity, rgba in MAWASHI_RGBA.items():
        recolor(spec, f"{entity}/base_link_visual", rgba)


def _add_ring(spec, radius: float = RING_RADIUS) -> None:
    """A dohyo, for the eye only: no geom here collides or is seen by a sensor.

    The ground stays the physics ground. Only its *drawn* extent is cut to the
    mound's square (a plane's size is rendering-only; its collision is infinite),
    so the mound's top is the ground the robots stand on and the hall floor is
    drawn a mound's height below it.
    """
    clay = [0.58, 0.42, 0.28, 1.0]           # darkened 10/10 night: under the hall's ring light the first shade read as pale yellow (the user)
    clay_side = [0.50, 0.36, 0.23, 1.0]
    # Straw and the sand band round it toned down with the clay (the user, 10/11: they
    # read as glaring white under the ring light).
    sand = [0.50, 0.41, 0.29, 1.0]
    straw = [0.46, 0.38, 0.22, 1.0]
    white = [0.84, 0.83, 0.80, 1.0]
    floor = [0.30, 0.24, 0.20, 1.0]
    vis = dict(contype=0, conaffinity=0, mass=0.0, group=DECOR_GROUP)

    def matte(name, rgba):
        """A material without MuJoCo's default shine: clay, sand and straw under the
        ring light glared white (the user, 10/11, twice)."""
        spec.add_material(name=name, rgba=rgba, specular=0.03, shininess=0.05, reflectance=0.0)
        return name

    for g in spec.geoms:  # the ground mjlab builds is the geom named `terrain`
        if g.name.endswith("terrain") and int(g.type) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            g.size = [_PLATFORM_HALF, _PLATFORM_HALF, g.size[2] or 0.05]
            g.material = matte("dohyo_clay", clay)
    h, P = _PLATFORM_HEIGHT, _PLATFORM_HALF
    spec.worldbody.add_geom(name="hall_floor", type=0, size=[8, 8, 0.05], pos=[0, 0, -h],
                            rgba=floor, **vis)
    # The mound's sides slope out to its foot, as a dohyo's do (a truncated pyramid;
    # until 10/11 they were drawn straight down, the user saw it).
    Pb = P + _MOUND_RUN
    top, bot = -0.0005, -h
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    verts = [(sx * P, sy * P, top) for sx, sy in corners] + [(sx * Pb, sy * Pb, bot) for sx, sy in corners]
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        quad = (i, j, 4 + j, 4 + i)
        faces += [(quad[0], quad[2], quad[1]), (quad[0], quad[3], quad[2]),   # both windings:
                  (quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])]   # seen from outside
    mound = spec.add_mesh()
    mound.name = "dohyo_mound_sides"
    mound.uservert = [c for v in verts for c in v]
    mound.userface = [k for f in faces for k in f]
    mound.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    spec.worldbody.add_geom(name="mound_sides", type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname="dohyo_mound_sides", material=matte("dohyo_side", clay_side), **vis)
    spec.worldbody.add_geom(name="janome_sand", type=5,
                            size=[radius + _BALE_RADIUS + _SAND_WIDTH, 0.0004, 0],
                            pos=[0, 0, 0.0004], material=matte("dohyo_sand", sand), **vis)
    spec.worldbody.add_geom(name="ring_clay", type=5, size=[radius - _BALE_RADIUS, 0.0006, 0],
                            pos=[0, 0, 0.0006], material="dohyo_clay", **vis)
    straw_mat = matte("dohyo_straw", straw)
    line_mat = matte("dohyo_line", white)

    def bale(name, cx, cy, along, half_len):
        """A straw bale, mostly buried, its axis along `along` (radians, in plane)."""
        c, s = math.cos(along), math.sin(along)
        # Capsule axis = local z; rotate +90 deg about the in-plane normal (-s, c).
        q = [math.cos(math.pi / 4), -s * math.sin(math.pi / 4), c * math.sin(math.pi / 4), 0.0]
        spec.worldbody.add_geom(name=name, type=3, size=[_BALE_RADIUS, half_len, 0],
                                pos=[cx, cy, _BALE_SHOWING - _BALE_RADIUS], quat=q,
                                material=straw_mat, **vis)

    # The circle: 20 slots of which the four on the axes are tokudawara, set out by a
    # bale's width -- 16 shobu-dawara plus 4.
    slots = 20
    half = radius * math.pi / slots * 0.92
    for i in range(slots):
        a = 2 * math.pi * i / slots
        on_axis = i % 5 == 0
        r = radius + (2 * _BALE_RADIUS if on_axis else 0.0)
        bale(f"{'toku' if on_axis else 'shobu'}_dawara_{i}", r * math.cos(a), r * math.sin(a),
             a + math.pi / 2, half)
    # Kaku-dawara along the mound's square edge, seven a side.
    edge = P - _BALE_RADIUS * 1.5
    for side, (ux, uy, nx, ny) in enumerate(((1, 0, 0, 1), (1, 0, 0, -1), (0, 1, 1, 0), (0, 1, -1, 0))):
        for k in range(7):
            t = -edge + (k + 0.5) * (2 * edge / 7)
            bale(f"kaku_dawara_{side}_{k}", ux * t + nx * edge, uy * t + ny * edge,
                 math.atan2(uy, ux), edge / 7 * 0.92)
    # Shikiri-sen: two white lines, 90 x 6 cm, 70 cm apart, scaled.
    for sign in (-1, 1):
        spec.worldbody.add_geom(
            name=f"shikiri_{'e' if sign < 0 else 'w'}", type=6,  # east (the robot) at -x
            size=[_LINE_WIDTH / 2, _LINE_LENGTH / 2, 0.0005],
            pos=[sign * (_LINE_GAP + _LINE_WIDTH) / 2, 0, 0.0012], material=line_mat, **vis)


def env_cfg(asset: Path | None = None, play: bool = False, gyoji: bool | None = None):
    """`gyoji`: the third Jumper refereeing in the ring; by default as
    JUMPER_SUMO_GYOJI=1 asks (replays), always for the gyoji's own training
    (`gyoji_policy/`)."""
    cfg = posture_env_cfg(asset=asset, play=play)

    # ── Two robots, facing each other, colliding with each other ──────────
    robot = cfg.scene.entities["robot"]
    opponent = copy.deepcopy(robot)
    gyoji_on = (os.environ.get("JUMPER_SUMO_GYOJI", "") == "1") if gyoji is None else gyoji
    gyoji_entity = copy.deepcopy(robot) if gyoji_on else None
    g_bit = _BIT_GYOJI if gyoji_on else 0
    robot.init_state.pos = (-START_X, 0.0, robot.init_state.pos[2])
    robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    opponent.init_state.pos = (START_X, 0.0, robot.init_state.pos[2])
    opponent.init_state.rot = (0.0, 0.0, 0.0, 1.0)  # yaw pi: facing -x
    robot.collisions = tuple(
        _with_channel(c, _BIT_ROBOT, _BIT_OPPONENT | g_bit) for c in robot.collisions)
    opponent.collisions = tuple(
        _with_channel(c, _BIT_OPPONENT, _BIT_ROBOT | g_bit) for c in opponent.collisions)
    cfg.scene.entities[OPPONENT] = opponent
    if gyoji_on:
        # The gyoji: a third Jumper beside the line, colliding with both (gyoji.py).
        # Put on its spot beside the line between the two (gyoji.WATCH_OFFSET), so
        # it stands still through the shikiri instead of walking there.
        gyoji_entity.init_state.pos = (0.0, 0.55, robot.init_state.pos[2])
        gyoji_entity.init_state.rot = (math.cos(-math.pi / 4), 0.0, 0.0, math.sin(-math.pi / 4))
        gyoji_entity.collisions = tuple(
            _with_channel(c, _BIT_GYOJI, _BIT_ROBOT | _BIT_OPPONENT) for c in gyoji_entity.collisions)
        cfg.scene.entities[GYOJI] = gyoji_entity
    # MuJoCo Warp sizes contact and constraint buffers per world. posture's 128 /
    # 512 are for one robot; before the reset places them, the two copies overlap
    # in the compiled model (327 contacts measured), and two robots in contact
    # need about twice the constraint rows of one.
    cfg.sim.nconmax = 512
    cfg.sim.njmax = 1536
    if gyoji_on:
        # Three robots overlapping before the reset places them: 654 contacts
        # measured (10/10 night: a Warp replay with the gyoji would not start).
        cfg.sim.nconmax = 1024
        cfg.sim.njmax = 2304

    # ── One ring for every environment ─────────────────────────────────────
    cfg.scene.env_spacing = 0.0
    previous = cfg.scene.spec_fn

    def spec_fn(spec):
        if previous is not None:
            previous(spec)
        _add_ring(spec)
        if play:
            # Replays only, so a training run's model is the same with or without it.
            _mawashi(spec)
        if gyoji_on:
            _dress(spec)
        if play and os.environ.get("JUMPER_SUMO_AUDIENCE") == "1":
            # Judges and spectators round the mound, as crabs (audience.py).
            from .audience import add_audience

            add_audience(spec, -_PLATFORM_HEIGHT, _PLATFORM_HALF)

    cfg.scene.spec_fn = spec_fn

    # ── Start positions: a little noise, still facing ──────────────────────
    start = {"x": (-0.02, 0.02), "y": (-0.03, 0.03), "z": (0.01, 0.03), "yaw": (-0.15, 0.15)}
    cfg.events["reset_base"].params["pose_range"] = dict(start)
    opp_base = copy.deepcopy(cfg.events["reset_base"])
    opp_base.params["asset_cfg"] = SceneEntityCfg(OPPONENT)
    cfg.events["reset_opponent_base"] = opp_base
    opp_joints = copy.deepcopy(cfg.events["reset_robot_joints"])
    opp_joints.params["asset_cfg"] = SceneEntityCfg(OPPONENT)
    cfg.events["reset_opponent_joints"] = opp_joints
    if "foot_friction" in cfg.events:
        friction = copy.deepcopy(cfg.events["foot_friction"])
        friction.params["asset_cfg"] = dataclasses.replace(
            friction.params["asset_cfg"], name=OPPONENT)
        cfg.events["opponent_foot_friction"] = friction
    # The other robot is the disturbance now.
    cfg.events.pop("push_robot", None)
    # Each bout starts in shikiri: both crouched, claws down (mdp.BoutState).
    cfg.events["reset_bout"] = EventTermCfg(func=reset_bout, mode="reset")
    # The opponent's claws are watched for the tachiai as the robot's are.
    for sensor in list(cfg.scene.sensors or ()):
        if sensor.name == "feet_ground_contact":
            opp_feet = copy.deepcopy(sensor)
            opp_feet.name = "opponent_feet_ground_contact"
            opp_feet.primary = dataclasses.replace(sensor.primary, entity=OPPONENT)
            cfg.scene.sensors = tuple(cfg.scene.sensors) + (opp_feet,)
            break

    # ── The learner's own joints: with the raised-hands shikiri (arms.py) ──
    joint_pos = cfg.actions["joint_pos"]
    cfg.actions["joint_pos"] = RitualJointPositionActionCfg(
        **{f.name: getattr(joint_pos, f.name) for f in dataclasses.fields(joint_pos) if f.init})

    # ── The opponent's commands and its frozen policy ──────────────────────
    cfg.commands[TWIST_OPP] = HeldCommandCfg(initial=(0.0, 0.0, 0.0))
    cfg.commands[POSTURE_OPP] = HeldCommandCfg(initial=(0.0, 0.0, 0.0, _NEUTRAL_HEIGHT))
    base = cfg.actions["joint_pos"]
    kwargs = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    kwargs["entity_name"] = OPPONENT
    # JUMPER_SUMO_OPPONENTS: a pool (comma-separated checkpoints) instead of the
    # official policy; JUMPER_SUMO_OPP_SPEED "lo,hi": a speed drawn per bout.
    pool = os.environ.get("JUMPER_SUMO_OPPONENTS") or str(OPPONENT_CHECKPOINT)
    speeds = os.environ.get("JUMPER_SUMO_OPP_SPEED")
    cfg.actions[OPPONENT] = OpponentPolicyActionCfg(
        **kwargs,
        checkpoint=pool,
        approach_speed_range=tuple(float(v) for v in speeds.split(",")) if speeds else None,
        observation_group=opponent_observation_group(cfg.observations["actor"]),
        neutral_height=_NEUTRAL_HEIGHT,
    )
    if gyoji_on:
        _add_gyoji_terms(cfg, kwargs)
    elif os.environ.get("JUMPER_SUMO_REFEREE", "") == "1":
        # The referee without the gyoji's body: kimarite counts for evaluations.
        _add_referee_terms(cfg)
        cfg.terminations["referee"] = TerminationTermCfg(
            func=referee_watch, params={"radius": RING_RADIUS})

    # ── The bout ends when either leaves the ring or falls ─────────────────
    fell = cfg.terminations["fell_over"]
    opp_fell = copy.deepcopy(fell)
    opp_fell.params["asset_cfg"] = SceneEntityCfg(OPPONENT)
    cfg.terminations["opponent_fell_over"] = opp_fell
    # Out: a foot on the ground beyond the straw -- the moment a toe touches
    # outside, as in a real bout (stepping on the straw is still in). Down: anything
    # but a foot on the clay. Both only once the bout has begun.
    edge = straw_edge(RING_RADIUS)
    for name, entity in (("robot_out", "robot"), ("opponent_out", OPPONENT)):
        cfg.terminations[name] = TerminationTermCfg(
            func=out_of_ring, params={"asset_cfg": SceneEntityCfg(entity), "edge": edge})
    for name, entity in (("robot_down", "robot"), ("opponent_down", OPPONENT)):
        cfg.terminations[name] = TerminationTermCfg(
            func=down_on_clay, params={"asset_cfg": SceneEntityCfg(entity)})
    for entity in ("robot", OPPONENT):
        cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (ContactSensorCfg(
            name=f"{entity}_body_ground",
            # What a Jumper stands and walks on is more than its six feet: the claw
            # palms and the calves touch the clay through ordinary walking (counted
            # 10/10: a palm in about half the steps of a bout). Down is the body or
            # an upper leg segment on the clay. Named body by body: a "subtree"
            # match's exclude only filters the subtree's root names, so the first
            # version (base_link's subtree minus the feet) counted every foot
            # contact, and every bout began with both wrestlers down -- dotai.
            primary=ContactMatch(mode="body", entity=entity, pattern=DOWN_BODIES),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found",), reduce="none", num_slots=1),)

    # ── Training: walk at the opponent, and be paid for the bout ───────────
    # Replay keeps play.py's operator command (keyboard / pad); a script that wants
    # the robot to walk at the opponent there installs `play_sumo.py`'s hook.
    if not play:
        cfg.commands["twist"] = ApproachCommandCfg()
        cfg.commands["posture"] = HeldCommandCfg(initial=(0.0, 0.0, 0.0, _NEUTRAL_HEIGHT))
        # The curricula widen command ranges this task no longer samples from.
        cfg.curriculum = {}
    # Rewards are scaled by the step time, so a one-step payment is divided by it.
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.rewards["sumo_win"] = RewardTermCfg(
        func=bout_outcome, weight=WIN_REWARD / step_dt,
        params={"terms": ("opponent_out", "opponent_fell_over", "opponent_down")})
    cfg.rewards["sumo_loss"] = RewardTermCfg(
        func=bout_outcome, weight=-LOSS_REWARD / step_dt,
        params={"terms": ("robot_out", "fell_over", "robot_down")})
    cfg.rewards["push_out"] = RewardTermCfg(func=push_out, weight=PUSH_OUT_WEIGHT)
    if gyoji_on:
        # With a gyoji the bout ends when it has called it, a moment after the
        # decision so the call can be seen (referee.bout_over); replay only.
        for name in ("fell_over", "opponent_fell_over", "robot_out", "opponent_out",
                     "robot_down", "opponent_down"):
            cfg.terminations.pop(name, None)
        for name in ("sumo_win", "sumo_loss"):
            cfg.rewards.pop(name, None)
        cfg.terminations["bout_over"] = TerminationTermCfg(
            func=bout_over, params={"radius": RING_RADIUS})
    if os.environ.get("JUMPER_SUMO_OPP_TACTICIAN"):
        # A frozen tactician on the west side instead of a walker (tactics/rival.py).
        from .tactics.rival import use_tactician_opponent

        use_tactician_opponent(cfg, RING_RADIUS)
    elif os.environ.get("JUMPER_SUMO_LEAGUE"):
        # A whole roster, one drawn every bout, walkers and tacticians alike.
        from .tactics.league import use_league_opponent

        use_league_opponent(cfg, RING_RADIUS, os.environ["JUMPER_SUMO_LEAGUE"])
    return cfg


def _add_gyoji_terms(cfg, action_kwargs: dict) -> None:
    """Commands, policy, resets and the referee's sensors for the gyoji."""
    cfg.commands[TWIST_GYOJI] = HeldCommandCfg(initial=(0.0, 0.0, 0.0))
    cfg.commands[POSTURE_GYOJI] = HeldCommandCfg(initial=(0.0, 0.0, 0.0, _NEUTRAL_HEIGHT))
    kwargs = dict(action_kwargs, entity_name=GYOJI)
    kwargs.update(
        # JUMPER_SUMO_GYOJI_WALKER: the gyoji's own legs (`jumper.gyoji_walk`, five
        # legs and the gunbai) instead of the six-legged official policy.
        checkpoint=os.environ.get("JUMPER_SUMO_GYOJI_WALKER") or str(OPPONENT_CHECKPOINT),
        observation_group=opponent_observation_group(
            cfg.observations["actor"], entity=GYOJI, twist=TWIST_GYOJI,
            posture=POSTURE_GYOJI, action=GYOJI),
        twist_name=TWIST_GYOJI, posture_name=POSTURE_GYOJI,
        neutral_height=_NEUTRAL_HEIGHT, radius=RING_RADIUS,
    )
    policy = os.environ.get("JUMPER_SUMO_GYOJI_POLICY", "")
    if policy:
        # A trained gyoji (`gyoji_policy/`, 10/14) instead of the scripted one: it
        # chooses where to stand and how to move from what it sees of the bout.
        from .gyoji_policy.mdp import LearnedGyojiActionCfg, gyoji_policy_group

        cfg.actions[GYOJI] = LearnedGyojiActionCfg(
            **kwargs, policy=policy, policy_group=gyoji_policy_group(RING_RADIUS))
    else:
        cfg.actions[GYOJI] = GyojiPolicyActionCfg(**kwargs)
    g_base = copy.deepcopy(cfg.events["reset_base"])
    g_base.params["asset_cfg"] = SceneEntityCfg(GYOJI)
    g_base.params["pose_range"] = {"z": (0.01, 0.03)}
    cfg.events["reset_gyoji_base"] = g_base
    g_joints = copy.deepcopy(cfg.events["reset_robot_joints"])
    g_joints.params["asset_cfg"] = SceneEntityCfg(GYOJI)
    cfg.events["reset_gyoji_joints"] = g_joints
    _add_referee_terms(cfg)


def _add_referee_terms(cfg) -> None:
    """The referee's reset and the contact sensors it judges with (referee.py)."""
    cfg.events["reset_referee"] = EventTermCfg(
        func=reset_referee, mode="reset", params={"radius": RING_RADIUS})
    east = ContactMatch(mode="subtree", pattern="base_link", entity="robot")
    west = ContactMatch(mode="subtree", pattern="base_link", entity=OPPONENT)
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
        ContactSensorCfg(name="bout_contact", primary=east, secondary=west,
                         fields=("found",), reduce="none", num_slots=1),
        # Body against body (base_link to base_link): yori rather than oshi.
        ContactSensorCfg(name="chest_contact",
                         primary=ContactMatch(mode="body", pattern="base_link", entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="base_link", entity=OPPONENT),
                         fields=("found",), reduce="none", num_slots=1),
    )
