"""Where the package's inputs live."""
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
ASSETS = PACKAGE / "robot" / "assets"
SKU2_DIR = ASSETS / "sku_2"
SKU2_XML = SKU2_DIR / "pupper.xml"
CONTRACT_JSON = SKU2_DIR / "sku2_contract.json"
MASSES_JSON = SKU2_DIR / "sku2_masses.json"
IMU_MOUNT_JSON = SKU2_DIR / "imu_mount.json"
SERVO_PROFILES = ASSETS / "servo_profiles.json"
ACTUATOR_PARAMS = PACKAGE / "actuators" / "params"
RUNS = REPO / "runs"


TEXT_SUFFIXES = {".xml", ".json", ".c", ".h", ".py", ".csv", ".md", ".txt"}


def sha256(path) -> str:
    """The file's digest; text sources are hashed with LF line endings, so a Windows checkout (core.autocrlf) gives
    the same digest as a Linux one."""
    p = Path(path)
    b = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


def rel(path) -> str:
    """A path relative to the package (for run configs), else as given."""
    p = Path(path).resolve()
    return str(p.relative_to(PACKAGE)).replace("\\", "/") if p.is_relative_to(PACKAGE) else str(p)
