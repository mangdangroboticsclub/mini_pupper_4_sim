"""Locations of this package's robot assets and of the files it generates.

The CAD, the screen, the weighed masses, the servo fits, and the policies
live under ``assets/`` in this repo. Nothing here reads a sibling checkout.
"""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]

ASSETS = REPO / "assets"
SKUS = ("sku_1", "sku_2")
ROLES = ("abd", "hip", "knee")
# One physics step. The T5 ub06 command delays are a few milliseconds, and MuJoCo
# integrates the same servos at this rate.
PHYSICS_DT = 0.001
_FIT = {"abd": "bam_t5ub06_id_3_m6.json", "hip": "bam_t5ub06_id_2_m6.json", "knee": "bam_t5ub06_id_3_m6.json"}

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


def contract_json(sku: str = "sku_2") -> Path:
    sku = require_sku(sku)
    if sku != "sku_2":
        raise ValueError("only sku_2 has a servo contract")
    return asset_dir(sku) / "sku2_contract.json"


def servo_map_json(sku: str = "sku_2") -> Path:
    return asset_dir(sku) / "servo_map.json"


def fit_json(role: str) -> Path:
    """The T5 ub06 m6 fit for one servo role: abd, hip, or knee."""
    try:
        name = _FIT[role]
    except KeyError:
        raise ValueError(f"unknown servo role {role!r}") from None
    return ASSETS / "actuators" / name


def policies() -> tuple[str, ...]:
    root = ASSETS / "policies"
    names = sorted(path.name for path in root.iterdir() if (path / "policy.onnx").is_file())
    if not names:
        raise FileNotFoundError(f"no policies under {root}")
    return tuple(names)


def policy_path(name: str = "flat_omni") -> Path:
    """A policy directory name, or a path to an ``.onnx`` file."""
    path = Path(name)
    if path.suffix == ".onnx":
        return path
    folder = ASSETS / "policies" / name / "policy.onnx"
    if not folder.is_file():
        raise FileNotFoundError(f"no policy {name!r}; available: {', '.join(policies())}")
    return folder


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
