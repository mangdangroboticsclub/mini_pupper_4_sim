"""The Mini Pupper v4 as an mjlab entity or a MuJoCo spec: a robot's ``pupper.xml`` with its weighed masses and
calibrated stand.

Changes, made on the MjSpec at load time (the file is never edited):

- the ghost robot (``g_base``), the ``stand`` keyframe, the twelve XML position actuators (the servo models replace
  them) and the XML sensor are removed; on SKU2 the visual-only accessories (lidar, Raspberry Pi, display, screen face:
  their files are not in the package) are removed; with ``strip_visuals`` the other mesh geoms (``contype =
  conaffinity = 0``) and their meshes too (masses are explicit ``<inertial>``s, so the dynamics do not change);
- an IMU at site ``imu``: gyro ``imu_ang_vel``, velocimeter ``imu_lin_vel``, accelerometer ``imu_lin_acc``;
- the weighed masses (the robot's ``measured/<robot>_masses*.json``: each module and each leg's thigh and calf scaled to
  their weighed masses, the base takes the rest, inertias scale with mass, the base's centre of mass set to the
  measured one);
- the feet get contact priority 1, so that their friction (randomised) is the one MuJoCo uses against the ground
  (equal priorities take the larger of the two, and the ground's is 1.0).

With ``limits`` the abduction (``*_hip``) and knee - thigh (``*_calf``) joints get the measured ranges (``limits``).

**Display collision** (``display_collision``, default off; SKU1 only): the SKU1's display (visual-only meshes) gets a
collision box on the base (``add_display_collision``, ``robots.Robot.display_box``) with the base box's contact
parameters, so it touches the floor and, with the self-collision options, the shins and feet (``leg_base_collisions``)
and the thighs (the get-up's thigh <-> base pair, ``envs.contact``) as the base box does. Refused on SKU2, which has no
box.

The stand (all twelve joints) comes from the robot's calibration (``calibration.Contract``), the firmware's neutral
pose.

**Robot, legs and feet** (``Legs``): ``robot`` is the body (``robots``: "sku2", the default, or "sku1", the short body
on the same legs). One leg set, "sku_2" (59.75 mm shin, the shin capsule 47.8 mm long, the foot site 0.04225 m along
the shin, the foot a mesh), with ``foot``:

- "sphere" (default): the foot geom replaced by a sphere of ``foot_radius_m`` (default 5.5 mm), placed so that at the
  stand (the calibration's ``stand_joint_rad``, the base level) its bottom is the mesh foot's lowest point, laterally
  at the mesh's mid-plane;
- "mesh": the model's foot mesh (``density=0``; MuJoCo collides its convex hull).

A policy's meta records them (``legs_meta``) with the model file's sha256; ``calibration.leg_set_name`` reads a leg
set's name ("new" is another name for "sku_2"). A SKU2 record has no ``robot`` key (a missing key means "sku2").
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mp4_sim import paths, robots
from mp4_sim.calibration import LEG_SETS, Contract, leg_set_name  # noqa: F401  (LEG_SETS re-exported)

LEGS = ("FR", "FL", "RR", "RL")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
LEG_COLLISION_GEOMS = tuple(f"{leg}_{part}_collision" for leg in ("FR", "FL", "RR", "RL") for part in ("thigh", "calf"))
BODY_CONTACT_BODIES = ("base",) + tuple(f"{leg}_hip" for leg in LEGS)   # the trunk and the motor cans
# The spawn height of the base origin, relative to the kinematic stand (``stand_geometry``): the feet start just inside
# the ground and settle (``init_base_z``).
SPAWN_DZ_M = -0.00045591091663704786
FOOT_RADIUS_M = 0.0055                  # the default foot sphere
FOOT_KINDS = ("sphere", "mesh")
# SKU2 pupper.xml's visual-only accessories (lidar, Raspberry Pi, display, screen face): their files are not in the
# package
ACCESSORY_MESHES = ("acc_display_housing", "acc_display_screen", "acc_lidar_mount", "acc_lidar_sensor", "acc_pi_black",
                    "acc_pi_metal", "acc_pi_pcb", "acc_pi_white", "base_screen_face")


@dataclass(frozen=True)
class Legs:
    """The robot, its legs and feet (module docstring)."""
    legs: str = "sku_2"
    foot: str = "sphere"
    foot_radius_m: float = FOOT_RADIUS_M
    robot: str = robots.DEFAULT

    def __post_init__(self):
        object.__setattr__(self, "legs", leg_set_name(self.legs))
        object.__setattr__(self, "robot", robots.get(self.robot).name)
        if self.foot not in FOOT_KINDS:
            raise ValueError(f"foot: one of {FOOT_KINDS}, not {self.foot!r}")
        if not 0.001 <= float(self.foot_radius_m) <= 0.02:
            raise ValueError(f"foot_radius_m {self.foot_radius_m} m: expected 1-20 mm")

    @property
    def spec(self) -> robots.Robot:
        return robots.get(self.robot)

    @property
    def xml(self) -> Path:
        return self.spec.xml

    def contract(self) -> Contract:
        """The robot's calibration."""
        return _contract(self.robot)

    def to_meta(self) -> dict:
        """What a policy's meta or a result file records (``robot`` only for a robot other than the default)."""
        out = {"legs": self.legs, "foot": self.foot,
               "foot_radius_m": float(self.foot_radius_m) if self.foot == "sphere" else None}
        if self.robot != robots.DEFAULT:
            out["robot"] = self.robot
        return out


def _contract(robot) -> Contract:
    return Contract.load(robots.get(robot).contract_json)


def legs_of(o=None, robot=None) -> Legs:
    """``Legs`` from options (an options dataclass, a run's options dict), a policy's meta (its ``legs`` block) or
    None (the defaults). A missing key means the default. ``robot``: the robot when ``o`` does not name one."""
    if o is None:
        return Legs(robot=robots.get(robot).name)
    if isinstance(o, Legs):
        return o
    if isinstance(o, dict) and isinstance(o.get("legs"), dict):
        o = o["legs"]
    get = o.get if isinstance(o, dict) else (lambda k, d=None: getattr(o, k, d))
    r = get("foot_radius_m", None)
    return Legs(leg_set_name(get("legs", None)), get("foot", None) or "sphere",
                FOOT_RADIUS_M if r is None else float(r), get("robot", None) or robots.get(robot).name)


def masses_path(name=None, robot=None) -> Path:
    """The masses file for the option ``name`` (``robots.Robot.masses_for``): None, the robot's own; a bare name, a
    file in the robot's ``measured/``; else a path."""
    r = robots.get(robot)
    name = r.masses_for(name)
    if name is None:
        return r.masses_path
    if (r.measured / str(name)).is_file():
        return r.measured / str(name)
    return paths.resolve(name)


def masses(path=None, robot=None) -> dict:
    return json.loads(masses_path(path, robot).read_text())


def apply_masses(spec: mujoco.MjSpec, m: dict) -> dict:
    new = {}
    for leg in LEGS:
        new[f"{leg}_hip"] = m["module_kg"]
        t, c = spec.body(f"{leg}_thigh").mass, spec.body(f"{leg}_calf").mass
        k = m["leg_kg"] / (t + c)
        new[f"{leg}_thigh"], new[f"{leg}_calf"] = t * k, c * k
    others = sum(b.mass for b in spec.bodies if b.name and b.name not in new and b.name not in ("base", "world"))
    new["base"] = m["total_kg"] - sum(new.values()) - others
    for name, mass in new.items():
        b = spec.body(name)
        b.fullinertia = np.asarray(b.fullinertia) * (mass / b.mass)
        b.mass = mass
    ipos = np.array(spec.body("base").ipos)
    ipos[0] = m["base_com_x_m"]
    if "base_com_y_m" in m:
        ipos[1] = m["base_com_y_m"]
    ipos[2] += m.get("base_com_dz_m", 0.0)   # a shift of the model's own base COM height
    spec.body("base").ipos = ipos
    return new


def leg_base_collisions(spec: mujoco.MjSpec) -> None:
    """Let each shin and foot collide with the base boxes (the model has no self-collision: robot geoms are contype 1,
    conaffinity 2). Bit 4 on the base's conaffinity and the calves' contype pairs exactly those; the thighs start on
    the base box and stay excluded. Terrain contacts are unchanged."""
    for g in spec.body("base").geoms:
        if g.contype or g.conaffinity:
            g.conaffinity |= 4
    for leg in LEGS:
        for g in spec.body(f"{leg}_calf").geoms:
            if g.contype or g.conaffinity:
                g.contype |= 4


DISPLAY_GEOM = "display_collision"
_BOX_CONTACT_FIELDS = ("contype", "conaffinity", "priority", "condim", "solmix", "solref", "solimp", "friction",
                       "margin", "gap", "group")


def display_box_frame(robot="sku1") -> tuple:
    """The display box of ``robot`` (``robots.Robot.display_box``): (centre (3,), quaternion wxyz (4,), half sizes (3,)),
    base frame, m. Refused for a robot without one."""
    r = robots.get(robot)
    if r.display_box is None:
        raise ValueError(f"robot {r.name} has no display collision box (display_collision is for "
                         f"{[x for x in robots.ROBOTS if robots.get(x).display_box is not None]})")
    pos, pitch_deg, half = r.display_box
    a = np.radians(float(pitch_deg)) / 2.0
    return np.asarray(pos, float), np.array([np.cos(a), 0.0, np.sin(a), 0.0]), np.asarray(half, float)


def add_display_collision(spec: mujoco.MjSpec, robot="sku1") -> None:
    """The display's collision box (``display_box_frame``) on ``base``, with a copy of the base box's contact parameters
    (call before the self-collision bits, which then treat it as part of the base)."""
    pos, quat, half = display_box_frame(robot)
    base = spec.body("base")
    body = next(g for g in base.geoms if g.contype or g.conaffinity)
    g = base.add_geom()
    g.name = DISPLAY_GEOM
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = half.tolist()
    g.pos = pos.tolist()
    g.quat = quat.tolist()
    g.density = 0.0
    g.rgba = [0.8, 0.2, 0.2, 0.4]
    for f in _BOX_CONTACT_FIELDS:
        setattr(g, f, getattr(body, f))


def stand_data(m: mujoco.MjModel, robot=None) -> mujoco.MjData:
    """The model at the robot's calibrated stand, the base at the origin and level, after ``mj_forward``."""
    c = _contract(robot)
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    d.qpos[3] = 1.0
    for name, v in zip(c.joint_order, c.stand_joint_rad):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    mujoco.mj_forward(m, d)
    return d


_stand_data = stand_data


def _mesh_vertices(m: mujoco.MjModel, geom: str) -> np.ndarray:
    """A mesh geom's vertices in its body's frame."""
    g = m.geom(geom)
    k = int(g.dataid[0])
    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, g.quat)
    return v @ R.reshape(3, 3).T + g.pos


@functools.lru_cache(maxsize=None)
def mesh_feet(robot: str = robots.DEFAULT) -> dict:
    """The foot meshes, per leg, in the calf's frame (m): bounding box, the lowest point at the stand (the
    base level; laterally at the mesh's mid-plane: the bottom is a ridge across the foot's width), the up direction there,
    and the mesh's size (vertices, faces, convex hull vertices)."""
    m = get_spec(legs=Legs(foot="mesh", robot=robot)).compile()
    d = stand_data(m, robot)
    out = {}
    for leg in LEGS:
        v = _mesh_vertices(m, f"{leg}_foot")
        up = d.xmat[m.body(f"{leg}_calf").id].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        low = v[int(np.argmin(v @ up))].copy()
        low[2] = 0.5 * (v[:, 2].min() + v[:, 2].max())
        k = int(m.geom(f"{leg}_foot").dataid[0])
        out[leg] = {"bbox_min": v.min(0), "bbox_max": v.max(0), "lowest": low, "up": up,
                    "vertices": int(m.mesh_vertnum[k]), "faces": int(m.mesh_facenum[k]),
                    "hull_vertices": int(m.mesh_graph[m.mesh_graphadr[k]]) if m.mesh_graphadr[k] >= 0 else None}
    return out


def foot_sphere_centre(leg: str, radius: float, robot: str = robots.DEFAULT) -> np.ndarray:
    """The sphere foot: its centre in the calf's frame, so that at the stand its bottom is the mesh foot's
    lowest point."""
    f = mesh_feet(robot)[leg]
    up = f["up"].copy()
    up[2] = 0.0                                    # the knee axis is horizontal at the stand (abduction 0)
    up /= np.linalg.norm(up)
    return f["lowest"] + radius * up


def _legs_and_visuals(spec: mujoco.MjSpec, legs: Legs, strip_visuals: bool) -> None:
    """The SKU2's accessories removed, the feet made ``legs.foot``, the visuals stripped. The SKU1 model has no accessory
    meshes; its screen face (material and texture) goes with the other visuals."""
    if legs.robot == "sku2":
        for g in list(spec.geoms):
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname in ACCESSORY_MESHES:
                spec.delete(g)
        for name in ACCESSORY_MESHES:
            spec.delete(spec.mesh(name))
    if legs.robot == "sku2" or strip_visuals:
        spec.delete(spec.material("face"))
        spec.delete(spec.texture("screen_face"))
    if legs.foot == "sphere":
        for leg in LEGS:
            g = spec.geom(f"{leg}_foot")
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.meshname = ""
            g.size = [float(legs.foot_radius_m), 0.0, 0.0]
            g.pos = foot_sphere_centre(leg, float(legs.foot_radius_m), legs.robot)
            g.quat = [1.0, 0.0, 0.0, 0.0]
    if strip_visuals:
        for g in list(spec.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
        keep = {f"{leg}_foot" for leg in LEGS} if legs.foot == "mesh" else set()
        for mesh in list(spec.meshes):
            if mesh.name not in keep:
                spec.delete(mesh)


def get_spec(strip_visuals: bool = True, masses_json: str | None = None,
             self_collision: bool = False, foot_solref: tuple | None = None, limits: bool = False,
             legs: Legs | None = None, display_collision: bool = False) -> mujoco.MjSpec:
    """``masses_json``: the masses file (``masses_path``; None: the robot's). ``limits``: the measured abduction and
    knee - thigh ranges as joint ranges (``limits.apply_limits``). ``legs``: the robot, its legs and feet (None: the
    defaults, SKU2). ``display_collision``: the display's collision box (``add_display_collision``; SKU1 only)."""
    legs = legs_of(legs)
    spec = mujoco.MjSpec.from_file(str(legs.xml))
    spec.delete(spec.body("g_base"))
    for key in list(spec.keys):
        spec.delete(key)
    for act in list(spec.actuators):
        spec.delete(act)
    for s in list(spec.sensors):
        spec.delete(s)
    _legs_and_visuals(spec, legs, strip_visuals)
    for name, kind in (("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
                       ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
                       ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    for leg in LEGS:                         # name the legs' collision capsules (feet are named in the MJCF)
        for part in ("thigh", "calf"):
            for k, g in enumerate(g for g in spec.body(f"{leg}_{part}").geoms
                                  if (g.contype or g.conaffinity) and g.name not in FOOT_GEOMS):
                g.name = f"{leg}_{part}_collision{k or ''}"
    for g in FOOT_GEOMS:
        spec.geom(g).priority = 1
        if foot_solref is not None:          # the feet's priority makes their contact parameters the ones used
            spec.geom(g).solref = list(foot_solref)
    apply_masses(spec, masses(masses_json, legs.robot))
    if display_collision:
        add_display_collision(spec, legs.robot)
    if self_collision:
        leg_base_collisions(spec)
    if limits:
        from mp4_sim.limits import apply_limits
        apply_limits(spec, robot=legs.robot)
    return spec


@functools.lru_cache(maxsize=None)
def stand_geometry(legs: Legs | None = None) -> dict:
    """At the robot's calibrated stand (``stand_joint_rad``), the base level, by forward kinematics (m): the base
    origin's height above flat ground (the feet's lowest points on it), each foot site's height above its foot's lowest
    point, the hip (thigh) axes' height above the feet's lowest points."""
    legs = legs_of(legs)
    m = get_spec(legs=legs).compile()
    d = stand_data(m, legs.robot)
    z0 = d.xpos[m.body("base").id][2]
    low, site, hip = {}, {}, {}
    for leg in LEGS:
        g = m.geom(f"{leg}_foot")
        if g.type[0] == mujoco.mjtGeom.mjGEOM_SPHERE:
            low[leg] = float(d.geom_xpos[g.id][2] - g.size[0])
        else:
            v = _mesh_vertices(m, g.name)
            body = int(m.geom_bodyid[g.id])
            low[leg] = float((v @ d.xmat[body].reshape(3, 3).T + d.xpos[body])[:, 2].min())
        site[leg] = float(d.site_xpos[m.site(f"{leg}_foot").id][2]) - low[leg]
        hip[leg] = float(d.xanchor[m.joint(f"{leg}_thigh").id][2]) - low[leg]
    ground = min(low.values())
    return {"base_z_m": float(z0 - ground), "foot_site_clearance_m": site, "hip_axis_height_m": hip,
            "foot_lowest_z_m": {k: v - ground for k, v in low.items()}}


def init_base_z(legs: Legs | None = None) -> float:
    """The spawn height of the base origin: the kinematic stand's (``stand_geometry``) plus ``SPAWN_DZ_M``."""
    return stand_geometry(legs_of(legs))["base_z_m"] + SPAWN_DZ_M


def robot_z_shift(legs: Legs | None = None) -> float:
    """How much lower (negative) or higher the robot's spawn height (``init_base_z``) is than SKU2's with the same legs
    and feet: 0.0 for sku2 (exactly), so a task's SKU2 height thresholds can move with the robot's own stand."""
    legs = legs_of(legs)
    if legs.robot == robots.DEFAULT:
        return 0.0
    return init_base_z(legs) - init_base_z(Legs(legs.legs, legs.foot, legs.foot_radius_m))


def foot_site_clearance(legs: Legs | None = None) -> float:
    """A foot site's height above the ground when its foot touches flat ground: the mean over the feet at the stand (the
    site sits on the shin's axis, the foot's lowest point behind and below it, so it changes a little with the shin's
    angle)."""
    legs = legs_of(legs)
    return float(np.mean(list(stand_geometry(legs)["foot_site_clearance_m"].values())))


def legs_meta(legs: Legs | None = None) -> dict:
    """The legs' block of a policy's meta, a run's config and an evaluation's result: the option and what it gives
    (for a robot other than the default, its provisional inputs, ``robots.Robot.provisional``). ``model`` is the model
    file relative to the package."""
    legs = legs_of(legs)
    g = stand_geometry(legs)
    extra = {} if legs.robot == robots.DEFAULT else {"robot_provisional": list(legs.spec.provisional)}
    return {**legs.to_meta(), **extra, "model": legs.xml.relative_to(paths.PACKAGE).as_posix(),
            "model_sha256": paths.sha256(legs.xml),
            "stand_base_z_m": round(g["base_z_m"], 6), "init_base_z_m": round(init_base_z(legs), 6),
            "foot_site_clearance_m": round(foot_site_clearance(legs), 6),
            "hip_axis_height_m": round(float(np.mean(list(g["hip_axis_height_m"].values()))), 6)}


def robot_cfg(actuators: tuple, strip_visuals: bool = True, masses_json: str | None = None,
              self_collision: bool = False, foot_solref: tuple | None = None, limits: bool = False,
              legs: Legs | None = None, display_collision: bool = False):
    """The robot as an mjlab ``EntityCfg``: spawned at ``init_base_z`` in the calibrated stand, its joints driven by
    ``actuators`` (``actuators.servo_model``), its spec from ``get_spec`` with these options."""
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
    legs = legs_of(legs)
    c = legs.contract()
    joint_pos = {name: float(v) for name, v in zip(c.joint_order, c.stand_joint_rad)}
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, init_base_z(legs)), joint_pos=joint_pos,
                                             joint_vel={".*": 0.0}),
        spec_fn=(lambda: get_spec(strip_visuals, masses_json, self_collision, foot_solref, limits, legs, True))
        if display_collision else
        (lambda: get_spec(strip_visuals, masses_json, self_collision, foot_solref, limits, legs)),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


def sources(legs: Legs | None = None, masses_json: str | None = None) -> dict:
    """Hashes of every input the model is built from (``paths.name`` -> sha256; written into each run's config)."""
    legs = legs_of(legs)
    files = [legs.xml, masses_path(masses_json, legs.robot)]
    if legs.foot == "mesh":
        files += [legs.xml.parent / "parts" / f"{leg}_foot.stl" for leg in LEGS]
    return {paths.name(f): paths.sha256(f) for f in files}
