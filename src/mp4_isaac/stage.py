"""Write the URDF the Isaac importer reads, and check that every mesh exists.

The repaired SKU2 URDF is the kinematic source. Visuals are the coloured pieces
from the MuJoCo model (``parts/`` and the materials in ``pupper.xml``), not one
colour on each link mesh. Collision stays the link STL. The screen is added as
two visual meshes in the base frame. Kinematics and masses are left as the CAD
export wrote them.
"""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .paths import asset_dir, display_json, require_sku, ros_package, staged_urdf, urdf_path

# The rgba values in sku_2/pupper.xml.
MATERIALS = {
    "yellow": "0.96 0.76 0.05 1",
    "black": "0.07 0.07 0.08 1",
    "metal": "0.55 0.56 0.58 1",
    "dark": "0.2 0.2 0.22 1",
}
LEG = {
    "front_right": "FR",
    "front_left": "FL",
    "back_right": "RR",
    "back_left": "RL",
}


def part_visuals(link_name: str) -> list[tuple[str, str]] | None:
    """MuJoCo visual pieces for one URDF link: ``(parts file, material)``."""
    if link_name == "base_link":
        return [("base_body.stl", "yellow"), ("base_frame.stl", "black"), ("base_detail.stl", "dark")]
    for prefix, code in LEG.items():
        if link_name == f"{prefix}_hip_link":
            return [
                (f"{code}_hip_motor.stl", "black"),
                (f"{code}_hip_flange.stl", "metal"),
                (f"{code}_hip_bracket.stl", "yellow"),
            ]
        if link_name == f"{prefix}_leg_upper_link":
            return [(f"{code}_thigh_a.stl", "yellow"), (f"{code}_thigh_b.stl", "dark")]
        if link_name == f"{prefix}_leg_lower_link":
            return [(f"{code}_calf.stl", "dark")]
    return None


def _material(name: str, rgba: str) -> ET.Element:
    mat = ET.Element("material", name=name)
    ET.SubElement(mat, "color", rgba=rgba)
    return mat


def _zero_inertial() -> ET.Element:
    inertial = ET.Element("inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0")
    ET.SubElement(
        inertial,
        "inertia",
        ixx="0",
        ixy="0",
        ixz="0",
        iyy="0",
        iyz="0",
        izz="0",
    )
    return inertial


def _visual(package: str, folder: str, mesh: str, material: str) -> ET.Element:
    visual = ET.Element("visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "mesh", filename=f"package://{package}/{folder}/{mesh}")
    ET.SubElement(visual, "material", name=material)
    return visual


def _fixed_joint(name: str, child: str) -> ET.Element:
    joint = ET.Element("joint", name=name, type="fixed")
    ET.SubElement(joint, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(joint, "parent", link="base_link")
    ET.SubElement(joint, "child", link=child)
    return joint


def build_tree(sku: str = "sku_2") -> ET.ElementTree:
    sku = require_sku(sku)
    urdf = urdf_path(sku)
    display_path = display_json(sku)
    package = ros_package(sku)
    if not urdf.is_file():
        raise FileNotFoundError(f"{sku} URDF not found: {urdf}")
    if not display_path.is_file():
        raise FileNotFoundError(f"display description not found: {display_path}")

    tree = ET.parse(urdf)
    robot = tree.getroot()
    for name, rgba in MATERIALS.items():
        robot.insert(0, _material(name, rgba))

    for link in robot.findall("link"):
        parts = part_visuals(link.get("name") or "")
        if parts is None:
            continue
        for visual in list(link.findall("visual")):
            link.remove(visual)
        for filename, material in parts:
            link.append(_visual(package, "parts", filename, material))

    display = json.loads(display_path.read_text())
    if display.get("body") != "base":
        raise ValueError(f"display.json attaches to {display.get('body')!r}, expected 'base'")
    for part in display["parts"]:
        name = part["mesh"]
        link = ET.Element("link", name=name)
        link.append(_zero_inertial())
        link.append(_visual(package, "display", part["file"], part["material"]))
        robot.append(link)
        robot.append(_fixed_joint(f"{name}_joint", name))
    robot.set("name", sku)
    return tree


def package_file(filename: str, sku: str) -> Path:
    """Resolve a mesh filename from the staged URDF against that SKU's directory."""
    package = ros_package(sku)
    prefix = f"package://{package}/"
    if not filename.startswith(prefix):
        raise ValueError(f"mesh path is not in the {package} package: {filename}")
    return asset_dir(sku) / filename[len(prefix):]


def mesh_paths(robot: ET.Element, sku: str) -> list[tuple[str, Path]]:
    found = []
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename")
        if filename:
            found.append((filename, package_file(filename, sku)))
    return found


def write(sku: str = "sku_2", path: Path | None = None) -> Path:
    sku = require_sku(sku)
    path = staged_urdf(sku) if path is None else path
    tree = build_tree(sku)
    missing = [str(p) for _, p in mesh_paths(tree.getroot(), sku) if not p.is_file()]
    if missing:
        raise FileNotFoundError("URDF meshes missing:\n" + "\n".join(missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def summary(robot: ET.Element, sku: str) -> str:
    links = [link.get("name") for link in robot.findall("link")]
    joints = [(j.get("name"), j.get("type")) for j in robot.findall("joint")]
    continuous = sum(1 for _, kind in joints if kind == "continuous")
    fixed = sum(1 for _, kind in joints if kind == "fixed")
    meshes = mesh_paths(robot, sku)
    lines = [
        f"links ({len(links)}): {', '.join(links)}",
        f"joints: {len(joints)} ({continuous} continuous, {fixed} fixed)",
        f"meshes: {len(meshes)}",
    ]
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage a SKU URDF for the Isaac importer")
    parser.add_argument("--sku", default="sku_2")
    args = parser.parse_args()
    path = write(args.sku)
    robot = ET.parse(path).getroot()
    print(f"wrote {path}")
    print(summary(robot, args.sku))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
