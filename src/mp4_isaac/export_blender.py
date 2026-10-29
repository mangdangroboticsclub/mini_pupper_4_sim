"""Write the stair crowd as a glTF file Blender can import.

The Isaac USD is a stack of physics payloads. This bakes the CAD parts at the
stand pose, one copy per robot, onto the blue treads.

    python -m mp4_isaac.export_blender
    python -m mp4_isaac.export_blender --crowd 24

In Blender: File → Import → glTF 2.0, and open the printed ``.glb``.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh

from mp4_sim.robot.mapping import ServoMap

from .paths import CACHE, asset_dir, servo_map_json
from .stage import MATERIALS, write as write_urdf
from .terrain import build, stair_formation

COLORS = {name: tuple(int(float(c) * 255) for c in rgba.split()[:3]) + (255,) for name, rgba in MATERIALS.items()}


def _floats(text: str, n: int) -> tuple[float, ...]:
    values = tuple(float(part) for part in text.split())
    if len(values) != n:
        raise ValueError(f"expected {n} numbers in {text!r}")
    return values


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = _rpy(*rpy)
    out[:3, 3] = xyz
    return out


def _axis_rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    x, y, z = np.asarray(axis, dtype=float)
    norm = np.linalg.norm((x, y, z))
    if norm == 0.0 or angle == 0.0:
        return np.eye(4)
    x, y, z = x / norm, y / norm, z / norm
    c, s = np.cos(angle), np.sin(angle)
    v = 1.0 - c
    out = np.eye(4)
    out[:3, :3] = [
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ]
    return out


def _mesh_path(filename: str) -> Path:
    prefix = "package://sku_2_urdf_fixed/"
    if not filename.startswith(prefix):
        raise ValueError(f"unexpected mesh path {filename}")
    return asset_dir("sku_2") / filename[len(prefix):]


def _painted(mesh: trimesh.Trimesh, rgba: tuple[int, int, int, int]) -> trimesh.Trimesh:
    painted = mesh.copy()
    painted.visual = trimesh.visual.ColorVisuals(mesh=painted, face_colors=np.tile(rgba, (len(painted.faces), 1)))
    return painted


def robot_parts(joint_angles: dict[str, float]) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    """Each visual mesh and its transform in the base frame, at ``joint_angles``."""
    root = ET.parse(write_urdf("sku_2")).getroot()
    visuals: dict[str, list[tuple[Path, np.ndarray, str]]] = {}
    for link in root.findall("link"):
        pieces = []
        for visual in link.findall("visual"):
            origin = visual.find("origin")
            xyz = _floats(origin.get("xyz", "0 0 0"), 3) if origin is not None else (0.0, 0.0, 0.0)
            rpy = _floats(origin.get("rpy", "0 0 0"), 3) if origin is not None else (0.0, 0.0, 0.0)
            mesh = visual.find("geometry/mesh")
            material = visual.find("material")
            if mesh is None:
                continue
            pieces.append((_mesh_path(mesh.get("filename")), _transform(xyz, rpy), material.get("name") if material is not None else "yellow"))
        visuals[link.get("name")] = pieces

    children: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)

    parts: list[tuple[trimesh.Trimesh, np.ndarray]] = []

    def walk(link: str, parent: np.ndarray) -> None:
        for path, local, material in visuals.get(link, []):
            parts.append((_painted(trimesh.load(path), COLORS[material]), parent @ local))
        for joint in children.get(link, []):
            origin = joint.find("origin")
            xyz = _floats(origin.get("xyz", "0 0 0"), 3) if origin is not None else (0.0, 0.0, 0.0)
            rpy = _floats(origin.get("rpy", "0 0 0"), 3) if origin is not None else (0.0, 0.0, 0.0)
            axis = joint.find("axis")
            axis_xyz = _floats(axis.get("xyz", "0 0 1"), 3) if axis is not None else (0.0, 0.0, 1.0)
            angle = 0.0 if joint.get("type") == "fixed" else joint_angles.get(joint.get("name"), 0.0)
            walk(joint.find("child").get("link"), parent @ _transform(xyz, rpy) @ _axis_rotation(axis_xyz, angle))

    walk("base_link", np.eye(4))
    return parts


def _box(center: tuple[float, float, float], size: tuple[float, float, float], color: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    rgba = tuple(int(channel * 255) for channel in color) + (255,)
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=np.tile(rgba, (len(mesh.faces), 1)))
    return mesh


def scene(crowd: int) -> trimesh.Scene:
    stand = {name: float(value) for name, value in zip(
        ("front_right_hip_joint", "front_right_leg_upper_joint", "front_right_leg_lower_joint",
         "front_left_hip_joint", "front_left_leg_upper_joint", "front_left_leg_lower_joint",
         "back_right_hip_joint", "back_right_leg_upper_joint", "back_right_leg_lower_joint",
         "back_left_hip_joint", "back_left_leg_upper_joint", "back_left_leg_lower_joint"),
        ServoMap(servo_map_json()).stand_joint_angles(),
    )}
    # ServoMap order is FR FL RR RL, which matches the tuple above. play.py uses LEG_ORDER the same way.
    parts = robot_parts(stand)
    lowest = min(float((mesh.vertices @ transform[:3, :3].T + transform[:3, 3])[:, 2].min()) for mesh, transform in parts)
    slots, _, _ = stair_formation(crowd)
    out = trimesh.Scene()
    for box in build("stairs", lanes=crowd).boxes:
        out.add_geometry(_box(box.center, box.size, box.color))
    for index, (x, y, surface) in enumerate(slots):
        place = np.eye(4)
        place[:3, 3] = (x, y, surface - lowest)
        for part, local in parts:
            out.add_geometry(part, transform=place @ local, node_name=f"robot_{index}_{id(local)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the stair crowd as a Blender glTF.")
    parser.add_argument("--crowd", type=int, default=24)
    parser.add_argument("--output", type=Path, default=CACHE / "stairs_crowd.glb")
    args = parser.parse_args()
    if args.crowd < 1:
        raise SystemExit("--crowd must be at least 1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene(args.crowd).export(args.output)
    print(f"wrote {args.output.resolve()}")
    print("Blender: File → Import → glTF 2.0")


if __name__ == "__main__":
    main()
