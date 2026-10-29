"""Convert the staged SKU2 URDF to a layered USD asset with Isaac Lab's importer.

Run inside the virtualenv from scripts/install.sh. This does not start
Omniverse Kit; Isaac Lab 3 loads the standalone importer when Isaac Sim is
absent.
"""

from __future__ import annotations

import argparse
import sys

from .paths import USD_DIR, asset_dir, require_sku, ros_package, staged_urdf
from .stage import write


def converter_cfg(sku: str, force: bool):
    from isaaclab.sim.converters import UrdfConverterCfg

    sku = require_sku(sku)
    gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
    drive = UrdfConverterCfg.JointDriveCfg(drive_type="force", target_type="none", gains=gains)
    return UrdfConverterCfg(
        asset_path=str(staged_urdf(sku)),
        usd_dir=str(USD_DIR),
        usd_file_name=f"{sku}.usd",
        force_usd_conversion=force,
        make_instanceable=True,
        physics_variant="physics",
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        collision_type="Convex Hull",
        collision_from_visuals=False,
        link_density=0.0,
        merge_mesh=False,
        robot_type="Quadruped",
        ros_package_paths=[{"name": ros_package(sku), "path": str(asset_dir(sku))}],
        joint_drive=drive,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )


def convert(sku: str = "sku_2", force: bool = False) -> str:
    sku = require_sku(sku)
    write(sku)
    USD_DIR.mkdir(parents=True, exist_ok=True)
    from isaaclab.sim.converters import UrdfConverter

    converter = UrdfConverter(converter_cfg(sku, force))
    return str(converter.usd_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a staged SKU URDF into USD")
    parser.add_argument("--sku", default="sku_2")
    parser.add_argument("--force", action="store_true", help="convert again even if a USD is already cached")
    args = parser.parse_args()
    print(f"usd: {convert(args.sku, force=args.force)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)
