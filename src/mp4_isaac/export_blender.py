"""Write the stair crowd as a glTF file Blender can import.

The Isaac USD is a stack of physics payloads. A still export bakes the stand
pose. The walking motion is recorded from the simulation and stored as glTF
node animation:

    python -m mp4_isaac.play --policy stairs --crowd 24 --blender

In Blender: File → Import → glTF 2.0, then press the spacebar.
"""

from __future__ import annotations

import argparse
import json
import struct
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


class Rig:
    """CAD visuals and the joint tree, loaded once."""

    def __init__(self) -> None:
        root = ET.parse(write_urdf("sku_2")).getroot()
        self.visuals: list[tuple[str, trimesh.Trimesh, np.ndarray]] = []
        grouped: dict[str, list[tuple[trimesh.Trimesh, np.ndarray]]] = {}
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
                name = material.get("name") if material is not None else "yellow"
                pieces.append((_painted(trimesh.load(_mesh_path(mesh.get("filename"))), COLORS[name]), _transform(xyz, rpy)))
            grouped[link.get("name")] = pieces
        self._children: dict[str, list[tuple[str, np.ndarray, tuple[float, float, float], str | None]]] = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = _floats(origin.get("xyz", "0 0 0"), 3) if origin is not None else (0.0, 0.0, 0.0)
            rpy = _floats(origin.get("rpy", "0 0 0"), 3) if origin is not None else (0.0, 0.0, 0.0)
            axis = joint.find("axis")
            axis_xyz = _floats(axis.get("xyz", "0 0 1"), 3) if axis is not None else (0.0, 0.0, 1.0)
            parent = joint.find("parent").get("link")
            name = None if joint.get("type") == "fixed" else joint.get("name")
            self._children.setdefault(parent, []).append((joint.find("child").get("link"), _transform(xyz, rpy), axis_xyz, name))
        self.grouped = grouped

    def link_poses(self, joint_angles: dict[str, float]) -> dict[str, np.ndarray]:
        """Link frames in the base frame."""
        poses: dict[str, np.ndarray] = {}

        def walk(link: str, parent: np.ndarray) -> None:
            poses[link] = parent
            for child, origin, axis, name in self._children.get(link, []):
                angle = 0.0 if name is None else float(joint_angles.get(name, 0.0))
                walk(child, parent @ origin @ _axis_rotation(axis, angle))

        walk("base_link", np.eye(4))
        return poses


def robot_parts(joint_angles: dict[str, float]) -> list[tuple[trimesh.Trimesh, np.ndarray]]:
    """Each visual mesh and its transform in the base frame, at ``joint_angles``."""
    rig = Rig()
    poses = rig.link_poses(joint_angles)
    parts = []
    for link, pieces in rig.grouped.items():
        for mesh, local in pieces:
            parts.append((mesh, poses[link] @ local))
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


def _quat_matrix(quat) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
            0.25 * scale,
        ])
    else:
        index = int(np.argmax([rotation[0, 0], rotation[1, 1], rotation[2, 2]]))
        if index == 0:
            scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            quat = np.array([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale])
        elif index == 1:
            scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            quat = np.array([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale])
        else:
            scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            quat = np.array([(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale])
    return quat / np.linalg.norm(quat)


def _trs(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix[:3, 3].astype(np.float32), _quat_from_matrix(matrix[:3, :3]).astype(np.float32)


class _Buffer:
    def __init__(self) -> None:
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def add(self, array: np.ndarray, accessor_type: str, component: int) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        data = np.ascontiguousarray(array)
        offset = len(self.blob)
        self.blob += data.tobytes()
        self.views.append({"buffer": 0, "byteOffset": offset, "byteLength": int(data.nbytes)})
        accessor = {
            "bufferView": len(self.views) - 1,
            "componentType": component,
            "count": int(data.shape[0]),
            "type": accessor_type,
        }
        flat = data.reshape(data.shape[0], -1)
        accessor["min"] = flat.min(axis=0).astype(float).tolist()
        accessor["max"] = flat.max(axis=0).astype(float).tolist()
        self.accessors.append(accessor)
        return len(self.accessors) - 1


def _pack_glb(payload: dict, blob: bytes) -> bytes:
    text = json.dumps(payload, separators=(",", ":")).encode()
    text += b" " * ((4 - len(text) % 4) % 4)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(text) + 8 + len(blob)
    header = struct.pack("<III", 0x46546C67, 2, total)
    return header + struct.pack("<I", len(text)) + b"JSON" + text + struct.pack("<I", len(blob)) + b"BIN\x00" + blob


def write_motion(path: Path, times: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray, joint_names: tuple[str, ...]) -> None:
    """Animated glTF. ``root_*`` are (frames, robots, …) and ``joint_pos`` is (frames, robots, 12)."""
    rig = Rig()
    course = build("stairs", lanes=root_pos.shape[1])
    buf = _Buffer()
    meshes = []
    materials = []
    mesh_of: dict[int, int] = {}

    def mesh_index(mesh: trimesh.Trimesh, color: tuple[int, int, int, int]) -> int:
        key = id(mesh)
        if key in mesh_of:
            return mesh_of[key]
        positions = mesh.vertices.astype(np.float32)
        indices = mesh.faces.reshape(-1).astype(np.uint32)
        primitive = {
            "attributes": {"POSITION": buf.add(positions, "VEC3", 5126)},
            "indices": buf.add(indices, "SCALAR", 5125),
            "material": len(materials),
        }
        materials.append({
            "pbrMetallicRoughness": {"baseColorFactor": [channel / 255 for channel in color[:3]] + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.8},
            "doubleSided": True,
        })
        meshes.append({"primitives": [primitive]})
        mesh_of[key] = len(meshes) - 1
        return mesh_of[key]

    nodes: list[dict] = []
    for box in course.boxes:
        color = tuple(int(channel * 255) for channel in box.color) + (255,)
        nodes.append({"mesh": mesh_index(_box(box.center, box.size, box.color), color), "name": "stair"})

    frames, robots, _ = root_pos.shape
    visual_nodes: list[tuple[int, int, str, np.ndarray]] = []
    for robot in range(robots):
        for link, pieces in rig.grouped.items():
            for mesh, local in pieces:
                color = tuple(int(mesh.visual.face_colors[0, channel]) for channel in range(4))
                node = len(nodes)
                nodes.append({"mesh": mesh_index(mesh, color), "name": f"robot_{robot}_{link}"})
                visual_nodes.append((node, robot, link, local))

    translations = [[] for _ in visual_nodes]
    rotations = [[] for _ in visual_nodes]
    for frame in range(frames):
        posed = []
        for robot in range(robots):
            angles = {name: float(joint_pos[frame, robot, joint]) for joint, name in enumerate(joint_names)}
            links = rig.link_poses(angles)
            root = np.eye(4)
            root[:3, :3] = _quat_matrix(root_quat[frame, robot])
            root[:3, 3] = root_pos[frame, robot]
            posed.append((root, links))
        for index, (_node, robot, link, local) in enumerate(visual_nodes):
            root, links = posed[robot]
            translation, rotation = _trs(root @ links[link] @ local)
            translations[index].append(translation)
            rotations[index].append(rotation)
    for sequence in rotations:
        for frame in range(1, frames):
            if float(np.dot(sequence[frame], sequence[frame - 1])) < 0.0:
                sequence[frame] = -sequence[frame]

    time_accessor = buf.add(np.asarray(times, dtype=np.float32), "SCALAR", 5126)
    channels = []
    samplers = []
    for index, (node, _robot, _link, _local) in enumerate(visual_nodes):
        translation = np.stack(translations[index]).astype(np.float32)
        rotation = np.stack(rotations[index]).astype(np.float32)
        for path_name, values, kind in (("translation", translation, "VEC3"), ("rotation", rotation, "VEC4")):
            samplers.append({"input": time_accessor, "output": buf.add(values, kind, 5126), "interpolation": "LINEAR"})
            channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path_name}})
    payload = {
        "asset": {"version": "2.0", "generator": "mp4-isaac"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "animations": [{"name": "walk", "channels": channels, "samplers": samplers}],
        "buffers": [{"byteLength": len(buf.blob)}],
        "bufferViews": buf.views,
        "accessors": buf.accessors,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pack_glb(payload, bytes(buf.blob)))
    print(f"wrote {path.resolve()}  ({frames} frames, {robots} robots)")
    print("Blender: File → Import → glTF 2.0, then press the spacebar.")


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
    print("This file is the stand pose. For the walk, run:")
    print("  python -m mp4_isaac.play --policy stairs --crowd 24 --blender")


if __name__ == "__main__":
    main()
