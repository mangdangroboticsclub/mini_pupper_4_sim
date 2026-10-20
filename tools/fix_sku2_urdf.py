"""
fix_sku2_urdf.py - turn the SolidWorks export of SKU 2 into a URDF build_model.py can read.

    python tools/fix_sku2_urdf.py                         # default export folder
    python tools/fix_sku2_urdf.py --export path/to/sku_2Full_assem_simplified_urdf

Reads the exporter's folder (``urdf/*.urdf`` + ``meshes/*.STL``) and writes

    src/mp4_sim/robot/assets/sku_2/urdf/sku_2_urdf_fixed.urdf
    src/mp4_sim/robot/assets/sku_2/meshes/<link>.STL

in exactly the shape of SKU 1's ``sku_1_urdf_fixed.urdf``: the same link and
joint names, the same body frame, one ``base_link``. Then
``python tools/build_model.py --sku sku_2`` builds the MuJoCo model from it.
You only need to run this again if the SKU 2 CAD is re-exported.

WHAT IS WRONG WITH THE EXPORT, AND WHAT THIS DOES ABOUT IT
  * body frame   the assembly is y-up and faces +x (display at the front, as in
                 the SolidWorks assembly): SKU 1's (x forward, y left, z up) is
                 CAD (x, -z, y). The body and the four hip modules are rotated
                 into SKU 1's frame and the origin is centred between the hips.
                 The CAD's leg names are right: fl -> front_left, and so on.
  * thighs, shins  exported with the thigh joint half a turn off: in the export
                 the legs point up, knees forward. They are re-posed by that
                 half turn about the thigh axis, which puts the knees backward
                 like SKU 1 (confirmed on the robot) and every thigh and shin
                 in exactly SKU 1's joint frame.
  * hip_fl       exported in the back-right hip's coordinate system
                 (``hip_br_sys`` in the exporter's CSV), so its frame sits on
                 the wrong hip and its mesh lands 273 mm behind the robot. The
                 mesh is moved onto the front-left hip (fr's x, bl's side) and
                 the joint is rebuilt by symmetry.
  * knees        lower_leg_br reused the thigh's coordinate system (origin 0 0 0),
                 lower_leg_fl's frame is off the knee axis, lower_leg_fr is
                 "fixed". All four knees are rebuilt at the thigh's knee bore
                 (35.36, 35.36 mm in the thigh frame - a 50 mm thigh, as SKU 1).
  * display, Pi  fixed placeholder boxes: folded into the base's mass and
                 inertia, their boxes dropped from the visuals.

Every joint gets SKU 1's orientation and axis for the same physical leg, so the
generated servo map has SKU 1's sign conventions. Meshes and inertials are
re-expressed in the repaired link frames (the exporter's placement of the body
and hip modules is kept, except hip_fl's as described above).

Checks (the script stops if one fails): the hip modules and the re-posed thighs
and shins line up with SKU 1's joint frames, the thighs' and shins' meshes sit
in their frames exactly as SKU 1's do (the knee axis is where the part's bore
is), and the four hips are symmetric.

Dependencies: trimesh, scipy (both in the mp4-sim environment).
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
EXPORT = os.path.join(ROOT, "cad", "sku_2", "sku_2Full_assem_simplified_urdf")
ASSETS = os.path.join(ROOT, "src", "mp4_sim", "robot", "assets")
OUT = os.path.join(ASSETS, "sku_2")
SKU1_URDF = os.path.join(ASSETS, "sku_1", "urdf", "sku_1_urdf_fixed.urdf")
NAME = "sku_2_urdf_fixed"

#: SKU 1 frame <- CAD frame: x forward = x_cad, y left = -z_cad, z up = y_cad
BODY = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0]])
#: the half turn about the lateral (thigh) axis the export's thighs and shins are off by
HALF_TURN = np.diag([-1.0, 1.0, -1.0])

#: CAD leg suffix -> SKU 1 link prefix
LEGS = {"fl": "front_left", "fr": "front_right", "bl": "back_left", "br": "back_right"}
#: the leg whose hip was exported in another leg's coordinate system
BROKEN_HIP = "fl"
#: knee bore in the thigh frame, m (the exporter's value on the legs it got right)
KNEE_XY = (-0.0353553390593255, 0.0353553390593252)
KNEE_Z = 0.00365            # + on the left legs, - on the right
#: hip -> thigh offset along the hip axis, m, where the export has no usable value
THIGH_OFFSET_FALLBACK = 0.012475
#: how far a check may be off, m
TOL = 2e-4


# =============================================================================
# reading
# =============================================================================
def rpy_matrix(rpy):
    return Rotation.from_euler("xyz", rpy).as_matrix()


def frame(rot, pos):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def read_urdf(path):
    root = ET.parse(path).getroot()
    links, joints = {}, {}
    for link in root.findall("link"):
        i = link.find("inertial")
        o = i.find("origin")
        ine = i.find("inertia").attrib
        I = np.array([[float(ine["ixx"]), float(ine["ixy"]), float(ine["ixz"])],
                      [float(ine["ixy"]), float(ine["iyy"]), float(ine["iyz"])],
                      [float(ine["ixz"]), float(ine["iyz"]), float(ine["izz"])]])
        R = rpy_matrix([float(v) for v in o.get("rpy").split()])
        links[link.get("name")] = dict(
            mass=float(i.find("mass").get("value")),
            com=np.array([float(v) for v in o.get("xyz").split()]),
            I=R @ I @ R.T)
    for j in root.findall("joint"):
        o = j.find("origin")
        a = j.find("axis")
        joints[j.find("child").get("link")] = dict(
            name=j.get("name"), type=j.get("type"), parent=j.find("parent").get("link"),
            T=frame(rpy_matrix([float(v) for v in o.get("rpy").split()]),
                    [float(v) for v in o.get("xyz").split()]),
            xyz=np.array([float(v) for v in o.get("xyz").split()]),
            rpy=[float(v) for v in o.get("rpy").split()],
            axis=np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None)
    return links, joints


def world_frames(joints, links):
    def T(link):
        if link not in joints:
            return np.eye(4)
        j = joints[link]
        return T(j["parent"]) @ j["T"]
    return {n: T(n) for n in links}


# =============================================================================
# the repair
# =============================================================================
def repair(export):
    links, joints = read_urdf(os.path.join(export, "urdf", "sku_2Full_assem_simplified_urdf.urdf"))
    meshes = {n: trimesh.load(os.path.join(export, "meshes", n + ".STL")) for n in links}
    W = world_frames(joints, links)

    # where each mesh really is, in CAD coordinates: the exporter's placement,
    # except the front-left chain, which goes onto the hip at fr's end, bl's side
    def world_mesh(n):
        return trimesh.transform_points(meshes[n].vertices, W[n])
    b_fl, b_fr, b_bl = (np.array([world_mesh(f"hip_{k}_link").min(0), world_mesh(f"hip_{k}_link").max(0)])
                        for k in ("fl", "fr", "bl"))
    shift = np.array([b_fr[0][0] - b_fl[0][0], 0.0, 0.0])
    assert np.allclose(b_fl[:, 1:], b_bl[:, 1:], atol=TOL), "hip_fl is not beside hip_bl"
    assert np.allclose(b_fl[:, 0] + shift[0], b_fr[:, 0], atol=TOL), "hip_fl does not match hip_fr"
    for n in (f"hip_{BROKEN_HIP}_link", f"upper_leg_{BROKEN_HIP}_link", f"lower_leg_{BROKEN_HIP}_link"):
        W[n] = frame(np.eye(3), shift) @ W[n]
    print(f"  hip_fl chain moved {shift[0] * 1000:+.1f} mm along CAD x")

    # CAD -> SKU 1 frame, centred between the hips
    hip_cad = {k: joints[f"hip_{k}_link"]["xyz"] for k in LEGS if k != BROKEN_HIP}
    hip_cad[BROKEN_HIP] = np.array([hip_cad["fr"][0], hip_cad["bl"][1], hip_cad["bl"][2]])
    hips = {k: BODY @ p for k, p in hip_cad.items()}
    centre = np.mean(list(hips.values()), axis=0)
    centre[2] = 0.0
    to_body = frame(BODY, -centre)           # SKU 1 frame <- CAD

    # SKU 1's joint orientations and axes, per physical leg
    _, sku1 = read_urdf(SKU1_URDF)
    sku1 = {j["name"].replace("back_riht_", "back_right_"): j for j in sku1.values()}

    def axis_of(name):
        return sku1[name]["axis"]

    out_links, out_joints = {}, []
    new_world = {"base_link": np.eye(4)}

    # ---- legs -------------------------------------------------------------
    for k, prefix in LEGS.items():
        left = prefix.endswith("left")
        hip_j, up_j, low_j = (f"{prefix}_hip_joint", f"{prefix}_leg_upper_joint",
                              f"{prefix}_leg_lower_joint")
        # hip
        T_hip = frame(rpy_matrix(sku1[hip_j]["rpy"]), hips[k] - centre)
        # thigh: offset along the hip axis, from the export where it is usable
        raw = joints[f"upper_leg_{k}_link"]["xyz"]
        sound = abs(raw[1]) < 1e-9 and abs(raw[2]) < 1e-9
        off = raw[0] if sound else (-THIGH_OFFSET_FALLBACK if left else THIGH_OFFSET_FALLBACK)
        T_up_rel = frame(rpy_matrix(sku1[up_j]["rpy"]), [off, 0.0, 0.0])
        # knee: at the thigh's knee bore
        T_low_rel = frame(np.eye(3), [KNEE_XY[0], KNEE_XY[1], KNEE_Z if left else -KNEE_Z])

        new_world[f"{prefix}_hip_link"] = T_hip
        new_world[f"{prefix}_leg_upper_link"] = T_hip @ T_up_rel
        new_world[f"{prefix}_leg_lower_link"] = T_hip @ T_up_rel @ T_low_rel
        out_joints += [
            (hip_j, "base_link", f"{prefix}_hip_link", T_hip[:3, 3], sku1[hip_j]["rpy"],
             axis_of(hip_j)),
            (up_j, f"{prefix}_hip_link", f"{prefix}_leg_upper_link", T_up_rel[:3, 3],
             sku1[up_j]["rpy"], axis_of(up_j)),
            (low_j, f"{prefix}_leg_upper_link", f"{prefix}_leg_lower_link", T_low_rel[:3, 3],
             [0.0, 0.0, 0.0], axis_of(low_j)),
        ]
        # the thigh and shin, turned half a turn about the hip's lateral axis
        hip = hips[k] - centre
        repose = frame(np.eye(3), hip) @ frame(HALF_TURN, [0.0, 0.0, 0.0]) @ frame(np.eye(3), -hip)
        for part, cad, place in (("hip_link", f"hip_{k}_link", to_body),
                                 ("leg_upper_link", f"upper_leg_{k}_link", repose @ to_body),
                                 ("leg_lower_link", f"lower_leg_{k}_link", repose @ to_body)):
            name = f"{prefix}_{part}"
            L = np.linalg.inv(new_world[name]) @ place @ W[cad]       # new link <- old link
            if part == "hip_link":
                # the module sits as the CAD mates it, a half turn from SKU 1's (it is
                # symmetric about the hip axis to within 3 mm)
                R = new_world[name][:3, :3] @ L[:3, :3] @ new_world[name][:3, :3].T
                assert np.allclose(R, HALF_TURN, atol=1e-3), f"{name}: hip module is not where expected"
            else:
                assert np.allclose(L[:3, :3], np.eye(3), atol=1e-3), f"{name}: frame rotation differs from SKU 1"
            out_links[name] = moved(links[cad], meshes[cad], L)
        print(f"  {prefix:12s} <- CAD {k}: hip at {fmt_mm(hips[k] - centre)} mm, "
              f"thigh offset {off * 1000:+.3f} mm{'' if sound else ' (rebuilt)'}")

    # ---- base: body + the fixed placeholder boxes ---------------------------
    parts = [moved(links[n], meshes[n], to_body @ W[n])
             for n in ("base_body_link", "display_link", "raspberry_link")]
    out_links["base_link"] = merge(parts, mesh=parts[0]["mesh"])
    return out_links, out_joints, centre


def moved(link, mesh, L):
    """A link's inertial and mesh, re-expressed through the transform L (new <- old)."""
    R, t = L[:3, :3], L[:3, 3]
    m = mesh.copy()
    m.apply_transform(L)
    return dict(mass=link["mass"], com=R @ link["com"] + t, I=R @ link["I"] @ R.T, mesh=m)


def merge(parts, mesh):
    M = sum(p["mass"] for p in parts)
    com = sum(p["mass"] * p["com"] for p in parts) / M
    I = np.zeros((3, 3))
    for p in parts:
        d = p["com"] - com
        I += p["I"] + p["mass"] * (d @ d * np.eye(3) - np.outer(d, d))
    return dict(mass=M, com=com, I=I, mesh=mesh)


def fmt_mm(v):
    return " ".join(f"{x * 1000:+.2f}" for x in v)


# =============================================================================
# checks against SKU 1
# =============================================================================
def check(out_links, out_joints):
    sku1_meshes = os.path.join(os.path.dirname(os.path.dirname(SKU1_URDF)), "meshes")
    worst = 0.0
    for prefix in LEGS.values():
        for part in ("leg_upper_link", "leg_lower_link"):
            ref = trimesh.load(os.path.join(sku1_meshes, f"{prefix}_{part}.STL")).bounds
            got = out_links[f"{prefix}_{part}"]["mesh"].bounds
            # in-plane (x, y) is the part's shape about its own pivot; z is the
            # stack-up along the axis, which SKU 2 is allowed to change
            err = float(np.abs(got[:, :2] - ref[:, :2]).max())
            worst = max(worst, err)
            assert err < 1e-3, f"{prefix}_{part}: mesh sits {err * 1000:.2f} mm off SKU 1's pivot"
    hips = np.array([xyz for name, _, _, xyz, _, _ in out_joints if name.endswith("_hip_joint")])
    asym = np.abs(np.abs(hips[:, :2]) - np.abs(hips[:, :2]).mean(0)).max()
    assert asym < TOL, f"hips are not symmetric ({asym * 1000:.2f} mm)"
    print(f"  checks: frames match SKU 1, thigh/shin pivots within {worst * 1000:.2f} mm of SKU 1's, "
          f"hips symmetric to {asym * 1000:.2f} mm")


# =============================================================================
# writing
# =============================================================================
def g(v):
    return " ".join(f"{x:.12g}" if abs(x) > 1e-15 else "0" for x in v)


def write(out_links, out_joints, out_dir):
    os.makedirs(os.path.join(out_dir, "urdf"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "meshes"), exist_ok=True)
    L = ['<?xml version="1.0" encoding="utf-8"?>',
         "<!-- GENERATED by tools/fix_sku2_urdf.py from the SolidWorks export of SKU 2 -",
         "     edit that, not this file. Same names and body frame as sku_1_urdf_fixed.urdf. -->",
         f'<robot name="{NAME}">']
    order = ["base_link"] + [c for _, _, c, _, _, _ in out_joints]
    for name in order:
        link = out_links[name]
        I = link["I"]
        link["mesh"].export(os.path.join(out_dir, "meshes", name + ".STL"))
        mesh = f"package://{NAME}/meshes/{name}.STL"
        L += [f'  <link name="{name}">',
              "    <inertial>",
              f'      <origin xyz="{g(link["com"])}" rpy="0 0 0"/>',
              f'      <mass value="{link["mass"]:.12g}"/>',
              (f'      <inertia ixx="{I[0, 0]:.12g}" ixy="{I[0, 1]:.12g}" ixz="{I[0, 2]:.12g}" '
               f'iyy="{I[1, 1]:.12g}" iyz="{I[1, 2]:.12g}" izz="{I[2, 2]:.12g}"/>'),
              "    </inertial>",
              f'    <visual><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh}"/></geometry></visual>',
              f'    <collision><origin xyz="0 0 0" rpy="0 0 0"/><geometry><mesh filename="{mesh}"/></geometry></collision>',
              "  </link>"]
    for name, parent, child, xyz, rpy, axis in out_joints:
        L += [f'  <joint name="{name}" type="continuous">',
              f'    <origin xyz="{g(xyz)}" rpy="{g(rpy)}"/>',
              f'    <parent link="{parent}"/>',
              f'    <child link="{child}"/>',
              f'    <axis xyz="{g(axis)}"/>',
              "  </joint>"]
    L.append("</robot>")
    path = os.path.join(out_dir, "urdf", NAME + ".urdf")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="SKU 2 SolidWorks export -> sku_2_urdf_fixed.urdf")
    ap.add_argument("--export", default=EXPORT, help=f"the exporter's folder (default: {EXPORT})")
    ap.add_argument("--out", default=OUT, help=f"model folder to write into (default: {OUT})")
    a = ap.parse_args(argv)
    print(f"reading {a.export}")
    out_links, out_joints, centre = repair(a.export)
    check(out_links, out_joints)
    total = sum(link["mass"] for link in out_links.values())
    print(f"  body origin moved {fmt_mm(centre)} mm (SKU 1 frame) to sit between the hips; "
          f"exported mass {total * 1000:.0f} g")
    path = write(out_links, out_joints, a.out)
    print("wrote", path, "+", os.path.join(a.out, "meshes"))
    print("now build it:  python tools/build_model.py --sku sku_2")


if __name__ == "__main__":
    main()
