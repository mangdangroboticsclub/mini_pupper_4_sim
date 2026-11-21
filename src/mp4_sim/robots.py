"""The robots this package models, their files and the files' recorded hashes.

- ``sku2``: Mini Pupper v4 SKU2, the long body (Raspberry Pi 5). Every default in the package is SKU2's.
- ``sku1``: Mini Pupper v4 SKU1, the short body on the same legs, servos and board.

A robot is chosen by passing it (a name or a ``Robot``) to the functions that build or load something; nothing in the
package keeps a current robot. Command-line tools add ``--robot`` with ``add_args`` and pass ``get(args.robot)`` on.

Each robot's files live in ``robot/assets/<sku>/``: the MJCF model (``pupper.xml``), the viewer scene, the servo map,
the meshes (``parts/``), the URDF, and ``measured/``: the calibration (``<robot>_contract.json``), the weighed masses,
the joint limits with their recorded sha256 and the hip envelope, the IMU mount and the measured sensor noise. A file
another robot's measurement stands in for is listed in ``provisional``.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

from mp4_sim import paths

ROBOTS = ("sku2", "sku1")
DEFAULT = "sku2"
# masses names the options may carry as a default: on a robot other than SKU2 they stand for its own (``masses_for``)
SKU2_DEFAULT_MASSES = (None, "sku2_masses.json")


@dataclass(frozen=True)
class Robot:
    name: str
    title: str
    dir: Path                         # robot/assets/<sku>
    masses_json: str                  # the weighed masses, a file in measured/
    contract_json: Path               # the calibration
    limits_json: Path                 # the joint limits; its sha256 recorded in <stem>.sha256 beside it
    hip_envelope_json: Path           # the hip's stop towards the body's middle as a function of the abduction
    imu_mount_json: Path
    noise_json: Path                  # the sensor noise measured on the robot
    top_z_m: float                    # the top of the base's collision boxes (base frame): an attachment sits on it
    lidar_x_mm: float                 # the optional lidar's axis, forward of the base origin
    lidar_mass_g: float = 43.2        # the lidar with its mount
    provisional: tuple = ()           # inputs another robot's measurement stands in for
    display_box: tuple | None = None  # the display's optional collision box (robot.model.add_display_collision):
                                      # (centre (m), pitch about the base's y (deg), half sizes (m)), base frame
    hashes: dict = field(default_factory=dict, compare=False, hash=False)   # file (relative to dir) -> sha256

    @property
    def xml(self) -> Path:
        return self.dir / "pupper.xml"

    @property
    def scene_xml(self) -> Path:
        return self.dir / "scene.xml"

    @property
    def servo_map_json(self) -> Path:
        return self.dir / "servo_map.json"

    @property
    def measured(self) -> Path:
        return self.dir / "measured"

    @property
    def limits_sha256_file(self) -> Path:
        return self.limits_json.with_suffix(".sha256")

    @property
    def masses_path(self) -> Path:
        return self.measured / self.masses_json

    def masses_for(self, name: str | None) -> str | None:
        """The masses file a build uses for the option ``name``: on SKU2 ``name`` itself; on another robot None and
        the SKU2 default name ``sku2_masses.json`` stand for the robot's own (with a warning for the latter); any
        other name or path loads as named."""
        if self.name == DEFAULT or name not in SKU2_DEFAULT_MASSES:
            return name
        if name is not None:
            warnings.warn(f"robot {self.name}: masses {name!r} is the SKU2 default; using {self.masses_json}",
                          stacklevel=2)
        return self.masses_json

    def files(self) -> dict:
        """Every recorded file (relative to ``dir``) -> its path."""
        return {k: self.dir / k for k in self.hashes}

    def to_meta(self) -> dict:
        return {"robot": self.name, "provisional": list(self.provisional),
                "contract_json": paths.name(self.contract_json), "limits_json": paths.name(self.limits_json)}


_SKU2_DIR = paths.ASSETS / "sku_2"
_SKU1_DIR = paths.ASSETS / "sku_1"

SKU2 = Robot(
    "sku2", "Mini Pupper v4 SKU2 (long body)", _SKU2_DIR, "sku2_masses.json",
    _SKU2_DIR / "measured" / "sku2_contract.json", _SKU2_DIR / "measured" / "sku2_joint_limits.json",
    _SKU2_DIR / "measured" / "sku2_hip_envelope.json", _SKU2_DIR / "measured" / "imu_mount.json",
    _SKU2_DIR / "measured" / "noise_measured.json", top_z_m=0.0265, lidar_x_mm=43.0,
    hashes={
        "SHA256SUMS": "e305cb05ad0ffe27327c0360d6e73035d4c6fb6d3594715f05d47dd738a428e8",     # the model files, meshes
        "pupper.xml": "455316472c9c059c7bb9e2d5236f87c59a6d58e42071c89507fcbe8bf83f3dcf",
        "scene.xml": "cf77a3dd2047fc94d957d08ba4fa50c1297fee9b33129a20a530c66eeb6c01ec",
        "servo_map.json": "cbd96e2550e5cbcd7a66eb1c784144aa19714b1a32246197131179f91c5b322c",
        "measured/sku2_contract.json": "86f1d330a9f30d7e2550d084e0593ef48d16ae4fc9970a9c32e2746f9cea7375",
        "measured/sku2_joint_limits.json": "9cad8e76f4b8bf505c4283905106be9786ef2c81a2bffa75ece404c51c026312",
        "measured/sku2_hip_envelope.json": "0d9d93b40598af6d0cc7feca381297077e9a500075d31de67f0848f3d9acac13",
        "measured/sku2_masses.json": "823884dc6eb1ad710395357013cc0b7971e303b1fb9d8def4579861c1800c00b",
        "measured/imu_mount.json": "0c1df51a25f121180858a5464ec15513c40e760552b1b15c9c80e80912845d87",
        "measured/noise_measured.json": "50a87c7b4802df21dc3a7a061161193a873ed2702365cf9341d9379e8af27d0e",
    })

# The SKU1's display (visual-only meshes base_display and base_screen in its model) as an optional collision box on the
# base: the smallest box about both meshes turned about the base's y axis (least volume at a pitch of 55.0 deg).
# (centre (m), pitch about the base's y (deg), half sizes (m)), base frame; every vertex of both meshes is inside
# (tests/test_display_collision.py).
SKU1_DISPLAY_BOX = ((0.07435, 0.00074, 0.00642), 55.0, (0.02042, 0.02732, 0.00505))

SKU1 = Robot(
    "sku1", "Mini Pupper v4 SKU1 (short body)", _SKU1_DIR, "sku1_masses_413g.json",
    _SKU1_DIR / "measured" / "sku1_contract.json", _SKU1_DIR / "measured" / "sku1_joint_limits.json",
    _SKU1_DIR / "measured" / "sku1_hip_envelope.json", _SKU2_DIR / "measured" / "imu_mount.json",
    _SKU2_DIR / "measured" / "noise_measured.json", top_z_m=0.0135, lidar_x_mm=-5.0, lidar_mass_g=47.2,
    provisional=("IMU mount and noise profile: SKU2's",),
    display_box=SKU1_DISPLAY_BOX,
    hashes={
        "SHA256SUMS": "b61eb53798d7f263d9e733fabd601b380c4c293f5233e48a37384e69376e015b",     # the model files, meshes
        "pupper.xml": "5ce84acc8ce53d1597d2f5df8c8ffc1ba3b56055e3f489c395cfc9211d2c2217",
        "scene.xml": "cf77a3dd2047fc94d957d08ba4fa50c1297fee9b33129a20a530c66eeb6c01ec",
        "servo_map.json": "fa53db99b5761ba62fb82d3aa4c54394f338875a9a6781f03ecdd10ad2dc29ed",
        "measured/sku1_contract.json": "48720bbdd8828779189aa5b1b2df7d0f01055870ea55639a31f8eaa1e9796bbe",
        "measured/sku1_masses_413g.json": "7c0a6fce4c23401ef5028a2d98678a07559cc6e55374d3b2f57ac27e2420e5d4",
        # the joint limits traced on this robot and their hip envelope; to replace them with a later fit, copy the new
        # files over these two (same names), write the limits' sha256 into sku1_joint_limits.sha256 and update both
        # hashes here
        "measured/sku1_joint_limits.json": "8b1274d545b5cb9f3e2222e2f2b2295961c877ea4400fdbb658a8550ba228dfc",
        "measured/sku1_hip_envelope.json": "c4cd8358e6d264558b675dee03587453df9a43ceda7f6d3d538e298025b51c9a",
    })

_BY_NAME = {r.name: r for r in (SKU2, SKU1)}


def get(robot: "str | Robot | None" = None) -> Robot:
    """The robot ``robot`` (a name or a ``Robot``; None: ``sku2``)."""
    if isinstance(robot, Robot):
        return robot
    name = robot or DEFAULT
    if name not in _BY_NAME:
        raise ValueError(f"robot: one of {ROBOTS}, not {name!r}")
    return _BY_NAME[name]


def add_args(parser, default_help: str = "sku2") -> None:
    """``--robot`` (``ROBOTS``; default None, read by ``get`` as SKU2)."""
    parser.add_argument("--robot", choices=ROBOTS, default=None,
                        help=f"the robot: sku2 (the long body) or sku1 (the short body) (default: {default_help})")


def check_files(robot: "str | Robot | None" = None) -> dict:
    """Each recorded file of the robot, and each file its ``SHA256SUMS`` lists (the model files and meshes) ->
    (expected sha256, actual sha256 or None if missing); raises ``ValueError`` on a difference."""
    r = get(robot)
    want_all = dict(r.hashes)
    sums = r.dir / "SHA256SUMS"
    if sums.is_file():
        for line in sums.read_text().splitlines():
            if line.strip():
                digest, rel = line.split(maxsplit=1)
                want_all.setdefault(rel.strip(), digest)
    out, bad = {}, []
    for rel, want in want_all.items():
        p = r.dir / rel
        have = paths.sha256(p) if p.is_file() else None
        out[rel] = (want, have)
        if have != want:
            bad.append(f"{rel}: {have} (recorded {want})")
    if bad:
        raise ValueError(f"robot {r.name}: files differ from the recorded hashes: " + "; ".join(bad))
    return out
