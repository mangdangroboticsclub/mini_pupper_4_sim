"""SKU2 as an mjlab entity: the generated model read in place, with SKU2's weighed masses and corrected stand.

Source: ``assets/sku_2/pupper.xml`` (from ``tools/build_model.py``; its hash is recorded in every run's config). Changes,
made on the MjSpec at load time (the file is never edited):

- the ghost robot (``g_base``), the ``stand`` keyframe, the twelve XML position actuators (the arm's actuators replace
  them) and the XML sensor are removed; with ``strip_visuals`` the mesh geoms (``contype = conaffinity = 0``) and the
  meshes too (masses are explicit ``<inertial>``s, so the dynamics do not change);
- an IMU at site ``imu``: gyro ``imu_ang_vel``, velocimeter ``imu_lin_vel``, accelerometer ``imu_lin_acc``;
- the weighed masses (assets/sku_2/sku2_masses.json: each module
  66.82 g, each leg's thigh and calf scaled together to 6.5 g, the base takes the rest of 450 g, inertias scale with
  mass, the base COM x set to the value solved for the measured 56.6 % rear share);
- the feet get contact priority 1, so that their friction (randomised) is the one MuJoCo uses against the ground
  (equal priorities take the larger of the two, and the ground's is 1.0);
- on rough ground (``rough``), the trunk's boxes and the hip cans meet the terrain through small spheres instead
  (``rough_ground_collision``);
- on the stairs (``stairs``), the feet get a stiffer contact (``stair_foot_contact``), so a foot pushed against
  a riser stays at its surface instead of creeping through the box.

The stand (all twelve joints) comes from ``assets/sku_2/sku2_contract.json``: the servo mapping with SKU2's stand
geometry corrected to the firmware's neutral pose (hip axes 70 mm above the feet).
"""
from __future__ import annotations

import json

import mujoco
import numpy as np

from .. import paths
from ..contract import Contract

LEGS = ("FR", "FL", "RR", "RL")
FOOT_GEOMS = tuple(f"{leg}_foot" for leg in LEGS)
FOOT_SITES = tuple(f"{leg}_foot" for leg in LEGS)
BODY_CONTACT_BODIES = ("base",) + tuple(f"{leg}_hip" for leg in LEGS)   # the trunk and the motor cans
# The base origin's height at the settled stand on the stock servos (71.9 mm); the start sits just above it.
INIT_BASE_Z = 0.0725
# The contact bit the rough-ground spheres collide on; the terrain's heightfield has it in its conaffinity, the robot's
# other geoms (contype 1, conaffinity 2) do not, so the spheres meet the ground and nothing else.
TRUNK_TERRAIN_BIT = 4
TERRAIN_CONAFFINITY = 1 | TRUNK_TERRAIN_BIT
# The feet's contact ``solref`` on the stairs (time constant s, damping ratio); MuJoCo's default is (0.02, 1). The
# feet have priority 1, so this governs every foot-ground pair. At 0.02 a foot that a servo holds against a riser at
# its 16 N limit creeps through the soft contact (the impedance leaves 5-10 % of the push unopposed) and is 8 mm past
# the riser plane within 0.1 s, far enough to be pushed out of the box's top onto the tread; at 0.005 (five physics
# steps) the same push is held at the riser's surface. Only the stairs task asks for it: the flat and rough runs, and
# the clones exported from them, keep the contact they were trained and checked with.
STAIR_FOOT_SOLREF = (0.005, 1.0)


def masses() -> dict:
    return json.loads(paths.MASSES_JSON.read_text())


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
    spec.body("base").ipos = ipos
    return new


def rough_ground_collision(spec: mujoco.MjSpec) -> None:
    """The trunk and the hip cans against a heightfield. mujoco_warp tests a heightfield against every prism under a
    geom's bounding box and stops at 50 (a warning per prism beyond, and the contacts dropped): at 10 mm cells that is
    a footprint of 5 x 5 cells, which the trunk's boxes (182 x 80 mm) and a tilted hip can (40 mm) exceed whenever
    they touch the ground (a fall, before ``body_contact`` ends the episode), and so does a thigh lying flat along
    the cells' diagonal (58 mm: a robot on its side lands on its thighs). So on rough ground those geoms no longer
    collide (contype 0; the robot's geoms never met each other: contype 1 against conaffinity 2) and others take
    their place, each spanning at most 4 x 4 cells: on the base, spheres of the lower box's thickness, bottoms flush
    with the box's bottom, in two rows of four (the upper box is only reachable past the tilt termination); in each
    hip, one sphere just inside the can (the body-contact sensor sees the spheres just the same); each thigh, two
    capsules of half its length. The calves (51 mm) stay: at most 5 x 5 cells, within the limit."""
    sphere = dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, contype=TRUNK_TERRAIN_BIT, conaffinity=0, group=4, density=0.0,
                  rgba=[0.8, 0.2, 0.2, 0.4])
    like = ("conaffinity", "group", "density", "rgba", "friction", "condim", "priority", "margin", "solref", "solimp")
    base = spec.body("base")
    boxes = [g for g in base.geoms if g.type == mujoco.mjtGeom.mjGEOM_BOX]
    for g in boxes:
        g.contype = 0
    low = min(boxes, key=lambda g: g.pos[2])
    r = float(low.size[2])
    for i, x in enumerate(np.linspace(-low.size[0] + r, low.size[0] - r, 4)):
        for j, y in enumerate((-low.size[1] + r, low.size[1] - r)):
            base.add_geom(name=f"base_ground_{i}{j}", size=[r, 0.0, 0.0], pos=[low.pos[0] + x, low.pos[1] + y, low.pos[2]],
                          **sphere)
    for leg in LEGS:
        hip = spec.body(f"{leg}_hip")
        cans = [g for g in hip.geoms if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER]
        assert len(cans) == 1, cans
        cans[0].contype = 0
        hip.add_geom(name=f"{leg}_hip_ground", size=[0.015, 0.0, 0.0], pos=list(cans[0].pos), **sphere)
        thigh = spec.body(f"{leg}_thigh")
        caps = [g for g in thigh.geoms if g.type == mujoco.mjtGeom.mjGEOM_CAPSULE]
        assert len(caps) == 1 and np.any(caps[0].fromto != 0.0), caps
        a, b = np.asarray(caps[0].fromto[:3]), np.asarray(caps[0].fromto[3:])
        for k, (p, q) in enumerate(((a, (a + b) / 2), ((a + b) / 2, b))):
            thigh.add_geom(name=f"{leg}_thigh_ground_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, contype=caps[0].contype,
                           size=list(caps[0].size), fromto=[*p, *q], **{key: getattr(caps[0], key) for key in like})
        caps[0].contype = 0


def stair_foot_contact(spec: mujoco.MjSpec) -> None:
    """A stiffer foot contact for the stairs (``STAIR_FOOT_SOLREF``): a riser is a wall, not a sponge."""
    for g in FOOT_GEOMS:
        spec.geom(g).solref = list(STAIR_FOOT_SOLREF)


def get_spec(strip_visuals: bool = True, rough: bool = False, stairs: bool = False) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(paths.SKU2_XML))
    spec.delete(spec.body("g_base"))
    for key in list(spec.keys):
        spec.delete(key)
    for act in list(spec.actuators):
        spec.delete(act)
    for s in list(spec.sensors):
        spec.delete(s)
    if strip_visuals:
        for g in list(spec.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
        for mesh in list(spec.meshes):
            spec.delete(mesh)
    for name, kind in (("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
                       ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
                       ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER)):
        spec.add_sensor(name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    for g in FOOT_GEOMS:
        spec.geom(g).priority = 1
    if rough:
        rough_ground_collision(spec)
    if stairs:
        stair_foot_contact(spec)
    apply_masses(spec, masses())
    return spec


def robot_cfg(actuators: tuple, strip_visuals: bool = True, rough: bool = False,
              stairs: bool = False):
    from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

    c = Contract.load()
    joint_pos = {name: float(v) for name, v in zip(c.joint_order, c.stand_joint_rad)}
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, INIT_BASE_Z), joint_pos=joint_pos,
                                             joint_vel={".*": 0.0}),
        spec_fn=lambda: get_spec(strip_visuals, rough, stairs),
        articulation=EntityArticulationInfoCfg(actuators=actuators),
    )


def sources() -> dict:
    """Hashes of every input the model is built from (written into each run's config)."""
    return {paths.rel(p): paths.sha256(p) for p in (paths.SKU2_XML, paths.SERVO_PROFILES, paths.MASSES_JSON)}
