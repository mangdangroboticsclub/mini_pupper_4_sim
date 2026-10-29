"""Report the staged URDF and, when a conversion exists, the USD it produced."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from .paths import USD_DIR, interface_usd, require_sku
from .stage import summary, write


def newest_usd() -> Path | None:
    if not USD_DIR.is_dir():
        return None
    layers = sorted(USD_DIR.rglob("*.usd*"), key=lambda path: path.stat().st_mtime)
    interfaces = [path for path in layers if path.parent == USD_DIR or path.name.endswith(".usda")]
    # Prefer an interface layer that sits at the root of a robot directory
    # (sku2.usda, or the importer's <robot>/<robot>.usda). Fall back to the
    # newest layer of any kind.
    rooted = [path for path in layers if path.suffix in {".usd", ".usda"} and "payloads" not in path.parts]
    chosen = rooted or interfaces or layers
    return chosen[-1] if chosen else None


def report_usd(path: Path) -> str:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"pxr could not open {path}")
    default = stage.GetDefaultPrim()
    links = []
    joints = []
    meshes = []
    mass = 0.0
    # Visual meshes live in instanceable geometry references, which a plain
    # traverse does not enter.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            links.append(prim.GetPath().pathString)
            mass_attr = prim.GetAttribute("physics:mass")
            if mass_attr and mass_attr.HasValue():
                mass += float(mass_attr.Get())
        if prim.IsA(UsdPhysics.Joint):
            joints.append(f"{prim.GetName()} ({prim.GetTypeName()})")
        if prim.IsA(UsdGeom.Mesh):
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
            meshes.append((prim.GetPath().pathString, 0 if counts is None else len(counts)))
    display = [name for name, _ in meshes if "display_bezel" in name or "display_panel" in name]
    empty = [name for name, faces in meshes if faces == 0]
    lines = [
        f"usd: {path}",
        f"default prim: {default.GetPath() if default else '(none)'}",
        f"physics variant: {(default.GetVariantSet('Physics').GetVariantSelection() if default else '')}",
        f"rigid bodies / articulation roots: {len(links)}",
        f"mass (kg): {mass:.6f}",
        f"joints: {len(joints)}",
        f"meshes: {len(meshes)} ({sum(faces for _, faces in meshes)} faces)",
        f"screen meshes: {len(display)}",
    ]
    if empty:
        lines.append("empty meshes: " + ", ".join(empty))
    if links:
        lines.append("bodies: " + ", ".join(links))
    if joints:
        lines.append("joint types: " + ", ".join(joints))
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Report a staged SKU and its USD")
    parser.add_argument("--sku", default="sku_2")
    args = parser.parse_args()
    sku = require_sku(args.sku)
    path = write(sku)
    print(f"staged: {path}")
    print(summary(ET.parse(path).getroot(), sku))
    usd = interface_usd(sku)
    if usd is None:
        print(f"usd: not converted yet (python -m mp4_isaac.convert --sku {sku})")
        return
    try:
        print(report_usd(usd))
    except ImportError:
        print(f"usd: {usd}")
        print("pxr is not installed in this interpreter; run inspect from the Isaac virtualenv")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
