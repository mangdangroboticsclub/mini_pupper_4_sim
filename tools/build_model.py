"""
build_model.py - turn the SolidWorks URDF into the MuJoCo model mp4_sim uses.

    python tools/build_model.py                      # SKU 2: writes src/mp4_sim/robot/assets/sku_2/
    python tools/build_model.py --sku sku_1          # SKU 1: writes src/mp4_sim/robot/assets/sku_1/
    python tools/build_model.py --model-dir /tmp/out # or somewhere else

You only need to run this again if the URDF or the STL meshes change; the
generated files are committed, so a normal checkout never runs it. For the
chosen SKU (see :mod:`mp4_sim.robot.sku`) it reads ``<model-dir>/urdf/<sku urdf>`` and
``<model-dir>/meshes`` and writes everything else under ``<model-dir>``:

    model/parts/*.stl      the meshes, split into coloured pieces
                           (display removed, motors black, body yellow)
    model/pupper.xml       the robot: 12 hinge joints named after the servos,
                           position servos, foot spheres, a ghost copy
    model/scene.xml        floor, lights, camera - what the teleop viewer opens
    model/servo_map.json   how a firmware servo angle becomes a joint angle

The URDF must use SKU 1's link names and body frame (x forward, y left, z up).
SKU 1's export does; SKU 2's raw export does not, and ``tools/fix_sku2_urdf.py``
converts it. What differs per SKU beyond the URDF is in ``SKU_BUILD``: how the
base mesh is coloured, the body's collision boxes, and the masses.

WHAT IT FIXES IN THE EXPORTED URDF
  * masses      the exporter weighed bare plastic (153 g in total for SKU 1).
                Scaled to the real robot: TOTAL_MASS, of which each of the four
                motor modules (the hip links) is MOTOR_MASS. Everything else
                keeps the exporter's proportions.
  * display     SKU 1: cut out of base_link.STL (the tilted frame at the front)
  * tiny parts  SKU 1: ~5600 sub-millimetre CAD bits (pins, text) dropped
  * joint names  named by servo: FR_hip(1) FR_thigh(2) FR_calf(3) ... and the
                "back_riht" typo is gone
  * contacts    the exporter's collision meshes are replaced with simple
                shapes: a body box, the four motor cans, the shins, and a
                sphere on each foot tip (fitted to the mesh: r = 4.9 mm)

THE KNEE
  The knee motor sits in the hip module and drives the shin through a rod,
  so a knee SERVO angle is the shin's angle to the vertical - not the angle
  between thigh and shin. The URDF models the knee as a plain hinge, so
  servo_map.json carries both, and :class:`mp4_sim.robot.mapping.ServoMap` converts.
  The rod itself is left out of the visuals: a free hinge can't move it
  correctly, so the knee is driven through a tendon instead.

Regenerating SKU 2's ``pupper.xml`` or ``servo_map.json`` invalidates
``sku2_contract.json``.

Dependencies: trimesh, scipy (both in the mp4-sim environment).
"""
import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir, "src")))

from mp4_sim.robot.sku import DEFAULT_SKU, SKUS, resolve_sku  # noqa: E402

#: Where the generated model is written, and where the CAD is read from: the
#: SKU's folder. ``--sku`` and ``--model-dir`` set them in :func:`main`.
SKU = resolve_sku(DEFAULT_SKU)
MODEL = str(SKU.dir)
URDF = str(SKU.urdf_path)
MESHES = os.path.join(MODEL, "meshes")
PARTS = os.path.join(MODEL, "parts")

# ---- the real robot (per SKU: see SKU_BUILD) -------------------------------
TOTAL_MASS = 0.360      # kg, whole robot with battery
MOTOR_MASS = 0.0655     # kg, each of the four motor modules (hip links)

# ---- servo model -----------------------------------------------------------
# A DC servo can't be infinitely strong or fast. Each joint gets:
#   stiffness  SERVO_KP        N*m per rad of position error
#   torque cap SERVO_TORQUE    N*m (stall torque; the current limit)
#   top speed  SERVO_SPEED     rad/s with no load (back-EMF as joint damping,
#                              so full torque at 0 speed, 0 torque at top speed)
# These are ESTIMATES, not measured. If the sim jumps higher or lower than
# the real robot, these three numbers are what to change.
SERVO_KP = 8.0
SERVO_TORQUE = 0.8          # N*m
SERVO_SPEED = 21.0          # rad/s  (~1200 deg/s, ~200 rpm)
SERVO_KV = 0.05             # N*m*s/rad - the servo's own D term (the AT32's
                            # kd_position); without it the joints ring and the
                            # walk shakes visibly

# ---- firmware leg model (main.c: L1, L2, NEUTRAL_Z) ------------------------
FW_L1, FW_L2, FW_NEUTRAL_Z = 50.0, 56.0, 70.0      # mm

# ---- colours ---------------------------------------------------------------
COLORS = {
    "yellow": (0.96, 0.76, 0.05, 1),
    "black":  (0.07, 0.07, 0.08, 1),
    "metal":  (0.55, 0.56, 0.58, 1),
    "dark":   (0.20, 0.20, 0.22, 1),
}

# leg name, servo ids (hip, thigh, knee), URDF prefix, side (+1 = left)
LEGS = [
    ("FR", (1, 2, 3),    "front_right", -1),
    ("FL", (4, 5, 6),    "front_left",  +1),
    ("RR", (7, 8, 9),    "back_right",  -1),
    ("RL", (10, 11, 12), "back_left",   +1),
]
# Firmware IK signs (main.c fRIK/fLIK/rRIK/rLIK):
#   right legs: thigh servo = -(th1-th1n), knee servo = +(th2-th2n)
#   left legs : thigh servo = +(th1-th1n), knee servo = -(th2-th2n)
IK_SIGN = {-1: (-1, +1), +1: (+1, -1)}

# Which way a HIP servo moves the foot, in firmware servo-angle terms
# (the angle servo_write() takes, + = bigger SCS number). +1 = foot toward
# the robot's LEFT. MEASURED with the firmware's Hip test (2026-09-28, seen
# from BEHIND the robot): +angle on a FRONT hip moves the foot LEFT, on a
# REAR hip RIGHT. (An earlier guess had all four the other way round - that
# made the sim's look left / right, turning and strafing mirror the robot.)
# If a hip moves the wrong way in the sim, flip it here (or in
# servo_map.json) - nothing else depends on it.
HIP_DIR = {"FR": +1, "FL": +1, "RR": -1, "RL": -1}


# =============================================================================
# URDF
# =============================================================================
def parse_urdf():
    root = ET.parse(URDF).getroot()
    links, joints = {}, {}
    for l in root.findall("link"):
        i = l.find("inertial")
        o = i.find("origin")
        ine = i.find("inertia").attrib
        links[l.get("name")] = dict(
            mass=float(i.find("mass").get("value")),
            com=[float(v) for v in o.get("xyz").split()],
            I=np.array([[float(ine["ixx"]), float(ine["ixy"]), float(ine["ixz"])],
                        [float(ine["ixy"]), float(ine["iyy"]), float(ine["iyz"])],
                        [float(ine["ixz"]), float(ine["iyz"]), float(ine["izz"])]]),
        )
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        name = j.get("name").replace("back_riht_", "back_right_")
        joints[name] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=[float(v) for v in o.get("xyz").split()],
            rpy=[float(v) for v in o.get("rpy").split()],
            axis=[float(v) for v in a.get("xyz").split()] if a is not None else [0, 0, 1],
        )
    return links, joints


def quat_wxyz(rpy):
    # URDF rpy = fixed-axis X, Y, Z  ->  scipy lower-case 'xyz' (extrinsic)
    x, y, z, w = Rotation.from_euler("xyz", rpy).as_quat()
    return [w, x, y, z]


def fmt(v, n=6):
    return " ".join(f"{x:.{n}g}" if abs(x) > 1e-12 else "0" for x in v)


# =============================================================================
# meshes -> coloured parts
# =============================================================================
def split(path):
    m = trimesh.load(path)
    return m.split(only_watertight=False)


def save(parts, name):
    if not parts:
        return None
    trimesh.util.concatenate(parts).export(os.path.join(PARTS, name + ".stl"))
    return name


def base_parts_sku1():
    """base_link.STL -> body (yellow), frame (black), detail (dark).
    The display is the tilted frame in front of the front cross-bar."""
    parts = split(os.path.join(MESHES, "base_link.STL"))
    spine = max(parts, key=lambda p: len(p.faces))
    lo, hi = spine.bounds
    body, frame, detail = [spine], [], []
    dropped = dict(display=0, tiny=0)
    for p in parts:
        if p is spine:
            continue
        b = p.bounds
        c = b.mean(0)
        diag = np.linalg.norm(p.extents)
        if b[0][0] > 0.062:                      # everything past x=62 mm: display
            dropped["display"] += 1
            continue
        if diag < 0.002:                         # sub-2 mm CAD bits
            dropped["tiny"] += 1
            continue
        inside = np.all(b[0] >= lo - 1e-4) and np.all(b[1] <= hi + 1e-4)
        if abs(c[0]) > 0.050 and not inside:     # front / rear cross-bars
            frame.append(p)
        elif not inside and p.extents[0] > 0.05 and p.extents[1] > 0.07:
            body.append(p)                       # top and bottom covers
        else:
            detail.append(p)                     # boards, battery, posts
    print(f"  base: removed display ({dropped['display']} part), "
          f"{dropped['tiny']} tiny parts")
    return [(save(body, "base_body"), "yellow"),
            (save(frame, "base_frame"), "black"),
            (save(detail, "base_detail"), "dark")]


def base_parts_sku2():
    """base_link.STL -> top plate on its posts, the band above the box and the
    bottom plate (yellow), the box and the end caps (black), the rest (dark).
    The display and the Pi are only placeholder boxes in this export;
    fix_sku2_urdf.py kept their mass and dropped their boxes, so there is
    nothing to cut out here."""
    parts = split(os.path.join(MESHES, "base_link.STL"))
    box = max(parts, key=lambda p: len(p.faces))
    body, frame, detail = [], [box], []
    for p in parts:
        if p is box:
            continue
        c = p.bounds.mean(0)
        if abs(c[0]) > 0.070:                         # front / rear end caps
            frame.append(p)
        elif p.extents[0] > 0.08 and p.extents[1] > 0.07:
            body.append(p)                            # top plate, band, bottom plate
        else:
            detail.append(p)                          # corner posts, pins
    print(f"  base: {len(body)} body, {len(frame)} frame, {len(detail)} detail parts")
    return [(save(body, "base_body"), "yellow"),
            (save(frame, "base_frame"), "black"),
            (save(detail, "base_detail"), "dark")]


#: What differs between SKUs beyond the URDF. ``body_boxes`` are the trunk's
#: collision boxes, ``(pos, half-size)`` in the base frame (MJCF strings),
#: fitted to the base mesh; masses in kg.
SKU_BUILD = {
    "sku_1": dict(base_parts=base_parts_sku1, total_mass=0.360, motor_mass=0.0655,
                  body_boxes=[(None, "0.072 0.040 0.0135")]),
    # SKU 1's masses: SKU 2 is marginally heavier but has not been weighed
    # (see mp4_sim.robot.sku.SKUS["sku_2"].notes)
    "sku_2": dict(base_parts=base_parts_sku2, total_mass=0.360, motor_mass=0.0655,
                  body_boxes=[("0 0 -0.0108", "0.091 0.040 0.0135"),         # box + end caps
                              ("0.012 0 0.0146", "0.059 0.0395 0.0119")]),   # top plate on its posts
}
assert set(SKU_BUILD) == set(SKUS), "every SKU needs a build profile"


def hip_parts(prefix, leg):
    """A hip link is one motor module: the 40 mm can (black), the output
    flange and shaft (metal), the mounting bracket (yellow)."""
    motor, metal, bracket = [], [], []
    for p in split(os.path.join(MESHES, f"{prefix}_hip_link.STL")):
        e = sorted(p.extents)
        c = p.bounds.mean(0)
        if e[1] > 0.038:                          # 40 mm diameter pieces
            motor.append(p)
        elif abs(c[1]) < 0.003 and abs(c[2]) < 0.003:
            metal.append(p)                       # on the motor axis
        else:
            bracket.append(p)
    return [(save(motor, f"{leg}_hip_motor"), "black"),
            (save(metal, f"{leg}_hip_flange"), "metal"),
            (save(bracket, f"{leg}_hip_bracket"), "yellow")]


def thigh_parts(prefix, leg):
    ps = sorted(split(os.path.join(MESHES, f"{prefix}_leg_upper_link.STL")),
                key=lambda p: -len(p.faces))
    # [0] main plate, [1] second plate, [2] the 2 mm knee rod (left out)
    return [(save(ps[:1], f"{leg}_thigh_a"), "yellow"),
            (save(ps[1:2], f"{leg}_thigh_b"), "dark")]


def calf_parts(prefix, leg):
    m = trimesh.load(os.path.join(MESHES, f"{prefix}_leg_lower_link.STL"))
    save([m], f"{leg}_calf")
    # foot tip: rounded end, fitted centre 55.25 mm along the shin, r 4.9 mm
    v = m.vertices[:, :2]
    far = v[np.argmax(np.linalg.norm(v, axis=1))]
    u = far / np.linalg.norm(far)
    u = np.array([math.copysign(1, u[0]), math.copysign(1, u[1])]) / math.sqrt(2)
    best = None
    for t in np.arange(0.050, 0.060, 0.00025):
        sel = (v @ u) > t
        if sel.sum() < 10:
            continue
        r = np.linalg.norm(v[sel] - t * u, axis=1)
        s = r[r > np.median(r)].std()
        if best is None or s < best[1]:
            best = (t, s, r.max())
    t, _, rad = best
    tip = [float(u[0] * t), float(u[1] * t), 0.0]
    return [(f"{leg}_calf", "dark")], tip, float(rad)


# =============================================================================
# MJCF
# =============================================================================
def inertial_xml(link, mass):
    k = mass / link["mass"]
    I = link["I"] * k
    return (f'<inertial pos="{fmt(link["com"])}" mass="{mass:.6g}" '
            f'fullinertia="{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} '
            f'{I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"/>')


def build_xml(links, joints, geoms, tips, tip_r, ghost, keyframe=None):
    """One robot (or its ghost). geoms: {link_key: [(mesh, color)]}"""
    P = "g_" if ghost else ""
    vis_cls = "ghost" if ghost else "visual"

    # masses: motor modules fixed, everything else scaled together
    other = [n for n in links if not n.endswith("_hip_link")]
    other_mass = sum(links[n]["mass"] for n in other)
    k_other = (TOTAL_MASS - 4 * MOTOR_MASS) / other_mass

    def mass_of(name):
        if ghost:
            return 1e-4
        return MOTOR_MASS if name.endswith("_hip_link") else links[name]["mass"] * k_other

    def vis(key, indent):
        out = []
        for mesh, col in geoms[key]:
            if mesh is None:
                continue
            mat = "ghost" if ghost else col
            out.append(f'{indent}<geom class="{vis_cls}" mesh="{mesh}" material="{mat}"/>')
        return "\n".join(out)

    L = []
    # the ghost is drawn, never simulated: gravcomp keeps it from falling
    # between the frames where the app places it
    gc = ' gravcomp="1"' if ghost else ""
    L.append(f'    <body name="{P}base" pos="0 0 0.2" childclass="pupper"{gc}>')
    L.append(f'      <freejoint name="{P}root"/>')
    L.append("      " + inertial_xml(links["base_link"], mass_of("base_link")))
    L.append(vis("base", "      "))
    if not ghost:
        for pos, size in SKU_BUILD[SKU.name]["body_boxes"]:
            at = f' pos="{pos}"' if pos else ""
            L.append(f'      <geom class="collision" type="box" size="{size}"{at}/>')
        L.append('      <site name="imu" pos="0 0 0" size="0.004"/>')
    for leg, ids, prefix, side in LEGS:
        jh = joints[f"{prefix}_hip_joint"]
        jt = joints[f"{prefix}_leg_upper_joint"]
        jk = joints[f"{prefix}_leg_lower_joint"]
        L.append(f'      <body name="{P}{leg}_hip" pos="{fmt(jh["xyz"])}" quat="{fmt(quat_wxyz(jh["rpy"]))}"{gc}>')
        L.append("        " + inertial_xml(links[jh["child"]], mass_of(jh["child"])))
        L.append(f'        <joint name="{P}{leg}_hip" class="{"gjoint" if ghost else "hip"}" axis="{fmt(jh["axis"])}"/>')
        L.append(vis(f"{leg}_hip", "        "))
        if not ghost:
            # motor can: 40 mm dia, axis along the hip link's x, x -16..21 mm
            L.append('        <geom class="collision" type="cylinder" size="0.020 0.0185" '
                     'pos="0.0027 0 0" quat="0.707107 0 0.707107 0"/>')
        L.append(f'        <body name="{P}{leg}_thigh" pos="{fmt(jt["xyz"])}" quat="{fmt(quat_wxyz(jt["rpy"]))}"{gc}>')
        L.append("          " + inertial_xml(links[jt["child"]], mass_of(jt["child"])))
        L.append(f'          <joint name="{P}{leg}_thigh" class="{"gjoint" if ghost else "thigh"}" axis="{fmt(jt["axis"])}"/>')
        L.append(vis(f"{leg}_thigh", "          "))
        if not ghost:
            L.append(f'          <geom class="collision" type="capsule" size="0.004" '
                     f'fromto="0 0 0 {fmt(jk["xyz"])}"/>')
        L.append(f'          <body name="{P}{leg}_calf" pos="{fmt(jk["xyz"])}" quat="{fmt(quat_wxyz(jk["rpy"]))}"{gc}>')
        L.append("            " + inertial_xml(links[jk["child"]], mass_of(jk["child"])))
        L.append(f'            <joint name="{P}{leg}_calf" class="{"gjoint" if ghost else "knee"}" axis="{fmt(jk["axis"])}"/>')
        L.append(vis(f"{leg}_calf", "            "))
        tip = tips[leg]
        if not ghost:
            shin_end = [tip[0] * 0.8, tip[1] * 0.8, 0]
            L.append(f'            <geom class="collision" type="capsule" size="0.0032" '
                     f'fromto="0 0 0 {fmt(shin_end)}"/>')
            L.append(f'            <geom class="foot" name="{leg}_foot" type="sphere" size="{tip_r:.4f}" pos="{fmt(tip)}"/>')
            L.append(f'            <site name="{leg}_foot" pos="{fmt(tip)}" size="0.003"/>')
        L.append("          </body>")
        L.append("        </body>")
        L.append("      </body>")
    L.append("    </body>")
    return "\n".join(x for x in L if x)


def write_robot(links, joints, geoms, tips, tip_r, stand=None, smap=None):
    meshes = sorted({m for gs in geoms.values() for m, _ in gs if m})
    asset = "\n".join(f'    <mesh name="{m}" file="{m}.stl"/>' for m in meshes)
    mats = "\n".join(f'    <material name="{k}" rgba="{fmt(v, 3)}" specular="0.3" shininess="0.4"/>'
                     for k, v in COLORS.items())
    names = [f"{leg}_{j}" for leg, *_ in LEGS for j in ("hip", "thigh", "calf")]
    tendons = ""
    act_lines = []
    for n in names:
        leg, j = n.split("_")
        if j == "calf" and smap is not None:
            # knee motor: the shin angle relative to the hip module
            #   = -axis_thigh * q_thigh - axis_knee * q_knee   (see pupper_map.py)
            act_lines.append(f'    <position name="{n}" tendon="{leg}_knee_drive" class="servo"/>')
        else:
            act_lines.append(f'    <position name="{n}" joint="{n}" class="servo"/>')
    acts = "\n".join(act_lines)
    if smap is not None:
        t = []
        for leg, *_ in LEGS:
            ax = smap["legs"][leg]["axis_sign"]
            t.append(f'    <fixed name="{leg}_knee_drive" damping="{SERVO_TORQUE / SERVO_SPEED:.5f}" armature="0.0002">\n'
                     f'      <joint joint="{leg}_thigh" coef="{-ax["thigh"]}"/>\n'
                     f'      <joint joint="{leg}_calf" coef="{-ax["knee"]}"/>\n'
                     f'    </fixed>')
        tendons = "  <tendon>\n" + "\n".join(t) + "\n  </tendon>\n"
    key = ""
    if stand is not None:
        qpos, ctrl = stand
        key = (f'  <keyframe>\n    <key name="stand" qpos="{fmt(qpos, 7)}" ctrl="{fmt(ctrl, 7)}"/>\n'
               f'  </keyframe>\n')
    xml = f"""<!-- GENERATED by build_model.py - edit that, not this file. -->
<mujoco model="pupper">
  <compiler angle="radian" meshdir="parts" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>

  <default>
    <default class="pupper">
      <!-- back-EMF: full torque at rest, none at SERVO_SPEED -->
      <joint damping="{SERVO_TORQUE / SERVO_SPEED:.5f}" armature="0.0002" frictionloss="0.003"/>
      <default class="hip"/>
      <default class="thigh"/>
      <!-- the knee hinge is passive: the knee motor drives the shin angle
           through the rod, modelled as a tendon (see <tendon>) -->
      <default class="knee">
        <joint damping="0.0005" armature="0.00002" frictionloss="0.001"/>
      </default>
      <default class="gjoint">
        <joint damping="0" armature="0" frictionloss="0"/>
      </default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="1" density="0"/>
      </default>
      <default class="ghost">
        <geom type="mesh" contype="0" conaffinity="0" group="3" density="0"/>
      </default>
      <default class="collision">
        <geom contype="1" conaffinity="2" group="4" density="0" rgba="0.8 0.2 0.2 0.4"/>
      </default>
      <default class="foot">
        <geom contype="1" conaffinity="2" group="4" density="0" friction="1.0 0.02 0.001"
              condim="3" rgba="0.2 0.8 0.2 0.6"/>
      </default>
    </default>
    <!-- servo: see SERVO_* in build_model.py. No ctrlrange and no joint
         range: every joint can turn all the way round, like the real servo
         with its calibration moved - the firmware's 0..1023 tick clamp is
         the only limit, and that is already in the stream. -->
    <default class="servo">
      <position kp="{SERVO_KP}" kv="{SERVO_KV}" forcerange="-{SERVO_TORQUE} {SERVO_TORQUE}"/>
    </default>
  </default>

  <asset>
{mats}
    <material name="ghost" rgba="0.25 0.65 1.0 0.28"/>
{asset}
  </asset>

  <worldbody>
{build_xml(links, joints, geoms, tips, tip_r, ghost=False)}
{build_xml(links, joints, geoms, tips, tip_r, ghost=True)}
  </worldbody>

{tendons}  <actuator>
{acts}
  </actuator>

  <sensor>
    <framequat name="body_quat" objtype="site" objname="imu"/>
  </sensor>
{key}</mujoco>
"""
    open(os.path.join(MODEL, "pupper.xml"), "w").write(xml)


SCENE = """<!-- GENERATED by build_model.py -->
<mujoco model="pupper scene">
  <include file="pupper.xml"/>
  <statistic center="0 0 0.08" extent="0.35"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-22" offwidth="1600" offheight="1000"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.35 0.45 0.55" rgb2="0.05 0.07 0.1" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" mark="edge" rgb1="0.22 0.24 0.27" rgb2="0.17 0.19 0.21"
             markrgb="0.45 0.45 0.45" width="300" height="300"/>
    <material name="floor" texture="grid" texuniform="true" texrepeat="10 10" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.2" dir="0 0 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="2" conaffinity="1"
          friction="1.0 0.02 0.001"/>
  </worldbody>
</mujoco>
"""


# =============================================================================
# servo -> joint mapping, from the model's own zero pose
# =============================================================================
def fw_neutral():
    ld = FW_NEUTRAL_Z
    th1n = -math.acos((FW_L1**2 + ld**2 - FW_L2**2) / (2 * FW_L1 * ld))
    th2n = math.asin((ld**2 - FW_L1**2 - FW_L2**2) / (2 * FW_L1 * FW_L2)) - th1n
    return th1n, th2n


def angle_from_down(v):
    """angle of a sagittal vector from straight down, + toward +x (forward)"""
    return math.atan2(v[0], -v[2])


def derive_map(xml_path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    th1n, th2n = fw_neutral()
    out = {
        "_about": ("Generated by build_model.py. Converts a firmware servo angle "
                   "(degrees, the angle servo_write() takes, 0 = stand pose, "
                   "calibration offset removed) into this model's joint angles. "
                   "See pupper_map.py."),
        "firmware": {"L1_mm": FW_L1, "L2_mm": FW_L2, "neutral_z_mm": FW_NEUTRAL_Z,
                     "th1n_deg": math.degrees(th1n), "th2n_deg": math.degrees(th2n)},
        "legs": {},
    }
    for leg, ids, prefix, side in LEGS:
        bt = m.body(f"{leg}_thigh").id
        bk = m.body(f"{leg}_calf").id
        sf = m.site(f"{leg}_foot").id
        pt, pk, pf = d.xpos[bt].copy(), d.xpos[bk].copy(), d.site_xpos[sf].copy()
        a1_0 = angle_from_down(pk - pt)
        a2_0 = angle_from_down(pf - pk)

        def axis_world(jname):
            j = m.joint(jname)
            return d.xmat[j.bodyid].reshape(3, 3) @ m.jnt_axis[j.id]
        nh, nt, nk = (axis_world(f"{leg}_{x}") for x in ("hip", "thigh", "calf"))
        # the axes must be (anti)parallel to x for the hip and y for the leg
        assert abs(abs(nh[0]) - 1) < 1e-3, (leg, nh)
        assert abs(abs(nt[1]) - 1) < 1e-3 and abs(abs(nk[1]) - 1) < 1e-3, (leg, nt, nk)
        s_ik_t, s_ik_k = IK_SIGN[side]
        out["legs"][leg] = {
            "servos": {"hip": ids[0], "thigh": ids[1], "knee": ids[2]},
            "joints": {"hip": f"{leg}_hip", "thigh": f"{leg}_thigh", "knee": f"{leg}_calf"},
            "ik_sign": {"thigh": s_ik_t, "knee": s_ik_k},
            "hip_dir": HIP_DIR[leg],
            "axis_sign": {"hip": int(round(nh[0])), "thigh": int(round(nt[1])),
                          "knee": int(round(nk[1]))},
            "zero_pose_deg": {"thigh": math.degrees(a1_0), "shin": math.degrees(a2_0)},
            "thigh_len_mm": float(np.linalg.norm(pk - pt) * 1000),
            "shin_len_mm": float(np.linalg.norm(pf - pk) * 1000),
        }
    return out


def main(argv=None):
    global SKU, MODEL, URDF, MESHES, PARTS, TOTAL_MASS, MOTOR_MASS
    ap = argparse.ArgumentParser(description="URDF -> MuJoCo model for mp4_sim")
    ap.add_argument("--sku", default=DEFAULT_SKU, choices=list(SKUS),
                    help=f"robot variant to build (default: {DEFAULT_SKU})")
    ap.add_argument("--model-dir", default=None,
                    help="directory holding urdf/ and meshes/ and receiving the generated "
                         "model (default: the SKU's folder, src/mp4_sim/robot/assets/<sku>)")
    a = ap.parse_args(argv)
    SKU = resolve_sku(a.sku)
    build = SKU_BUILD[SKU.name]
    TOTAL_MASS, MOTOR_MASS = build["total_mass"], build["motor_mass"]
    MODEL = os.path.abspath(a.model_dir or str(SKU.dir))
    URDF = os.path.join(MODEL, "urdf", SKU.urdf)
    MESHES = os.path.join(MODEL, "meshes")
    PARTS = os.path.join(MODEL, "parts")

    os.makedirs(PARTS, exist_ok=True)
    print(f"{SKU.title}: {URDF}")
    for note in SKU.notes:
        print(f"  NOTE: {note}")
    links, joints = parse_urdf()
    tot = sum(l["mass"] for l in links.values())
    print(f"URDF: {len(links)} links, {len(joints)} joints, exported mass {tot*1000:.0f} g")
    for n, j in joints.items():
        if j["type"] == "fixed":
            print(f"  WARNING: {n} is fixed in the URDF - made a hinge here")

    print("splitting meshes ...")
    geoms = {"base": build["base_parts"]()}
    tips, tip_r = {}, None
    for leg, ids, prefix, side in LEGS:
        geoms[f"{leg}_hip"] = hip_parts(prefix, leg)
        geoms[f"{leg}_thigh"] = thigh_parts(prefix, leg)
        geoms[f"{leg}_calf"], tips[leg], tip_r = calf_parts(prefix, leg)
    print(f"  foot tip: centre {np.linalg.norm(tips['FR'])*1000:.2f} mm from the knee, "
          f"radius {tip_r*1000:.2f} mm")

    # pass 1: model without keyframe -> derive the mapping from its zero pose
    write_robot(links, joints, geoms, tips, tip_r)
    open(os.path.join(MODEL, "scene.xml"), "w").write(SCENE)
    smap = derive_map(os.path.join(MODEL, "scene.xml"))
    json.dump(smap, open(os.path.join(MODEL, "servo_map.json"), "w"), indent=2)
    for leg, v in smap["legs"].items():
        print(f"  {leg}: thigh {v['thigh_len_mm']:.2f} mm, shin {v['shin_len_mm']:.2f} mm, "
              f"zero pose thigh {v['zero_pose_deg']['thigh']:+.1f} deg, "
              f"shin {v['zero_pose_deg']['shin']:+.1f} deg, axes {v['axis_sign']}")

    # pass 2: add the stand keyframe (all servos 0), feet on the floor
    import mujoco

    from mp4_sim.robot.rig import GhostRig
    rig = GhostRig(mujoco.MjModel.from_xml_path(os.path.join(MODEL, "scene.xml")),
                   os.path.join(MODEL, "servo_map.json"))
    m = rig.model
    d = mujoco.MjData(m)
    q = rig.map.servo_to_joint(np.zeros(12))
    rig.set_qpos(d, q)
    mujoco.mj_forward(m, d)
    lowest = min(d.geom_xpos[m.geom(f"{leg}_foot").id][2] for leg, *_ in LEGS) - tip_r
    d.qpos[2] -= lowest
    d.qpos[rig.groot + 2] = d.qpos[2]         # the ghost root follows
    qpos = d.qpos.copy()
    write_robot(links, joints, geoms, tips, tip_r, smap=smap)       # + knee tendons
    m = mujoco.MjModel.from_xml_path(os.path.join(MODEL, "scene.xml"))
    d = mujoco.MjData(m)
    rig = GhostRig(m, os.path.join(MODEL, "servo_map.json"))
    rig.set_ctrl(d, q)
    write_robot(links, joints, geoms, tips, tip_r, stand=(qpos, d.ctrl.copy()), smap=smap)
    m = mujoco.MjModel.from_xml_path(os.path.join(MODEL, "scene.xml"))
    robot_mass = sum(m.body_mass[i] for i in range(m.nbody)
                     if not m.body(i).name.startswith("g_") and i > 0)
    print(f"model: {m.nbody-1} bodies, {m.njnt} joints, {m.nu} servos, "
          f"robot mass {robot_mass*1000:.0f} g, stand height {qpos[2]*1000:.1f} mm")
    print("wrote", os.path.join(MODEL, "scene.xml"), "+", os.path.join(MODEL, "pupper.xml"),
          "+", os.path.join(MODEL, "servo_map.json"), "+", PARTS)
    if SKU.name == "sku_2":
        print("NOTE: sku2_contract.json was derived from the previous pupper.xml / servo_map.json")


if __name__ == "__main__":
    main()
