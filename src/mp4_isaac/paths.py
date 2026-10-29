"""Locations of the SKU2 source assets and of the files this package generates."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
WORKSPACE = REPO.parent

MP4_SIM = WORKSPACE / "mp4-sim"
ASSETS = MP4_SIM / "src" / "mp4_sim" / "robot" / "assets"
SKUS = ("sku_1", "sku_2")

CACHE = REPO / ".cache"
USD_DIR = CACHE / "usd"


def require_sku(name: str) -> str:
    sku = name.strip().lower().replace("-", "_").replace(" ", "_")
    if sku in ("1", "2"):
        sku = f"sku_{sku}"
    if sku not in SKUS:
        raise ValueError(f"unknown SKU {name!r}; expected sku_1 or sku_2")
    return sku


def asset_dir(sku: str = "sku_2") -> Path:
    return ASSETS / require_sku(sku)


def ros_package(sku: str = "sku_2") -> str:
    """The URDF's package:// name. The asset directory is that package root."""
    return f"{require_sku(sku)}_urdf_fixed"


def urdf_path(sku: str = "sku_2") -> Path:
    sku = require_sku(sku)
    return asset_dir(sku) / "urdf" / f"{ros_package(sku)}.urdf"


def display_json(sku: str = "sku_2") -> Path:
    return asset_dir(sku) / "display" / "display.json"


def staged_urdf(sku: str = "sku_2") -> Path:
    return CACHE / "urdf" / f"{require_sku(sku)}_with_display.urdf"


def interface_usd(sku: str) -> Path | None:
    """The converted interface layer for one SKU, if convert has been run."""
    sku = require_sku(sku)
    matches = sorted(USD_DIR.glob(f"**/{sku}_with_display.usd*"), key=lambda path: path.stat().st_mtime)
    rooted = [path for path in matches if "payloads" not in path.parts and path.suffix in {".usd", ".usda"}]
    return rooted[-1] if rooted else None


# SKU2 remains the default the earlier modules import by name.
SKU2 = asset_dir("sku_2")
URDF = urdf_path("sku_2")
DISPLAY_JSON = display_json("sku_2")
ROS_PACKAGE = ros_package("sku_2")
STAGED_URDF = staged_urdf("sku_2")

VENV = Path.home() / ".venvs" / "mp4-isaac"
