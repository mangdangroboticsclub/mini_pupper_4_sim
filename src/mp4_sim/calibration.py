"""The servo calibration of a Mini Pupper v4: the stand pose, the servo signs and offsets, and the firmware's units.

numpy only, and no import from the rest of ``mp4_sim``: a robot's runtime can copy this file (with the package's
``__init__.py``) and nothing else.

Coordinates. Per servo id 1..12 (FR, FL, RR, RL x abduction, hip, knee), a target or a reading is the servo's own
angle in radians, relative to the stand: ``rel_i = sign_i * radians(servo_deg_i)``, where ``servo_deg`` is the angle
the firmware's ``servo_write`` takes without the calibration offset (0 = the stand). In the MuJoCo model the abduction
and hip are the ``*_hip`` and ``*_thigh`` joints and the knee is the ``*_knee_drive`` tendon (the shin's angle); the
signs, the stand and the tendon coefficients are in the robot's calibration file (``<robot>_contract.json``).

The firmware's units:
``tick = 511 + (int)((servo_deg + offset) * 10240 / 3100)``, ``dd = (int)(3100 - tick * 3100 / 1024)``, both in
float32. The exact inverse for a measured AT32 angle is ``servo_deg = (DD0 - dd) / 10 - offset``, DD0 the dd of tick
511 (convert with 1024, not 1023).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_PACKAGE = Path(__file__).resolve().parent
# Each robot's calibration file in the package (``robots.Robot.contract_json`` names the same files).
CONTRACT_FILES = {"sku2": "robot/assets/sku_2/measured/sku2_contract.json",
                  "sku1": "robot/assets/sku_1/measured/sku1_contract.json"}
DEFAULT_ROBOT = "sku2"
CONTRACT_JSON = _PACKAGE / CONTRACT_FILES[DEFAULT_ROBOT]

DD0 = 3100.0 - 511.0 * 3100.0 / 1024.0          # 1553.02734375, the dd of tick 511
ABDUCTION = (0, 3, 6, 9)                        # indices of the abductions in servo order
ACTION_SIZE = 12                                # servos
LEG_NAMES = ("FR", "FL", "RR", "RL")            # servo order of the legs
LEG_SETS = ("sku_2",)                           # the robot model's leg sets (``robot.model.Legs``)
LEG_ALIASES = {"new": "sku_2"}                  # another name policy files may use for a leg set
ABD_OUT_SIGN = (1.0, -1.0, -1.0, 1.0)           # per leg (FR, FL, RR, RL): the abduction's model / rel sign of "outward"
TRACK_DD = (0.0, 3100.0)                        # the potentiometer's electrical track (dd); a ~50 deg dead zone beyond
F32 = np.float32


def contract_path(robot=None) -> Path:
    """A robot's calibration file in the package: ``robot`` a name (``"sku1"``, ``"sku2"``), an object with a
    ``contract_json`` attribute (``robots.Robot``), or None (SKU2)."""
    if robot is None:
        return CONTRACT_JSON
    p = getattr(robot, "contract_json", None)
    if p is not None:
        return Path(p)
    if robot not in CONTRACT_FILES:
        raise ValueError(f"robot: one of {tuple(CONTRACT_FILES)}, not {robot!r}")
    return _PACKAGE / CONTRACT_FILES[robot]


@dataclass
class Contract:
    signs: np.ndarray
    stand_rel_rad: np.ndarray            # the stand in the model's coordinates (joint angles, tendon lengths)
    stand_joint_rad: np.ndarray          # all twelve model joints at the stand (hip, thigh, calf per leg)
    offsets_deg: np.ndarray
    stand_dd: np.ndarray
    stand_ticks: np.ndarray
    joint_order: list
    policy_targets: list
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None, robot=None) -> "Contract":
        """The calibration file ``path``; None: the package's file of ``robot`` (None: SKU2)."""
        p = contract_path(robot) if path is None else Path(path)
        d = json.loads(p.read_text())
        c = cls(signs=np.array(d["signs"], dtype=float), stand_rel_rad=np.array(d["stand_policy_rad"]),
                stand_joint_rad=np.array(d["stand_joint_rad"]), offsets_deg=np.array(d["offsets_deg"]),
                stand_dd=np.array(d["stand_dd"]), stand_ticks=np.array(d["stand_ticks"]),
                joint_order=d["joint_order"], policy_targets=d["policy_targets"], meta=d)
        return c

    # ---- firmware units -------------------------------------------------------------------------------------------
    def ticks_from_servo_deg(self, servo_deg) -> np.ndarray:
        ang = (np.asarray(servo_deg, dtype=F32) + self.offsets_deg.astype(F32)).astype(F32)
        sig = 511 + np.trunc(ang * F32(10240.0) / F32(3100.0)).astype(np.int64)
        return np.clip(sig, 0, 1023)

    @staticmethod
    def dd_from_ticks(ticks) -> np.ndarray:
        t = np.asarray(ticks, dtype=F32)
        return np.trunc(F32(3100.0) - t * F32(3100.0) / F32(1024.0)).astype(np.int64)

    def servo_deg_from_dd(self, dd) -> np.ndarray:
        """A measured AT32 angle (dd) in servo degrees. Not a target's inverse: the dd of tick t reads up to 0.1 deg
        above t's angle, and C's truncation towards zero turns that into t + 1 for ticks below 511."""
        return (DD0 - np.asarray(dd, dtype=float)) / 10.0 - self.offsets_deg

    def servo_deg_of_tick(self, ticks) -> np.ndarray:
        """The target angle (servo degrees) whose firmware tick is exactly ``ticks``: the centre of the tick's
        truncation interval (k = tick - 511: [k, k+1) for k > 0, (k-1, k] for k < 0, (-1, 1) for k = 0)."""
        k = np.asarray(ticks, dtype=float) - 511.0
        return (k + 0.5 * np.sign(k)) * 3100.0 / 10240.0 - self.offsets_deg

    # ---- servo-relative coordinates -------------------------------------------------------------------------------
    def rel_from_servo_deg(self, servo_deg) -> np.ndarray:
        return self.signs * np.radians(np.asarray(servo_deg, dtype=float))

    def servo_deg_from_rel(self, rel) -> np.ndarray:
        return np.degrees(np.asarray(rel, dtype=float)) * self.signs

    def rel_from_ticks(self, ticks) -> np.ndarray:
        """A target that the firmware path turns into exactly ``ticks`` (the targets that reproduce a firmware gait's
        ticks)."""
        return self.rel_from_servo_deg(self.servo_deg_of_tick(ticks))

    def rel_from_dd(self, dd) -> np.ndarray:
        """A measured AT32 angle (STATE ``pos_dd``) as a position (rad, relative to the stand)."""
        return self.rel_from_servo_deg(self.servo_deg_from_dd(dd))

    def quantise_rel(self, rel) -> tuple[np.ndarray, np.ndarray]:
        """(the dd the firmware path would send for a target, that dd back in servo-relative coordinates)."""
        dd = self.dd_from_ticks(self.ticks_from_servo_deg(self.servo_deg_from_rel(rel)))
        return dd, self.rel_from_dd(dd)


def leg_set_name(name: str | None) -> str:
    """A leg set's name (``LEG_SETS``), an alias read as its leg set; None: the first."""
    name = LEG_ALIASES.get(name, name) if name else LEG_SETS[0]
    if name not in LEG_SETS:
        raise ValueError(f"legs: one of {LEG_SETS}, not {name!r}")
    return name


def contract_tendon_coefs(contract: "Contract") -> np.ndarray:
    """Per leg (FR, FL, RR, RL) the knee tendon's coefficients on the thigh and calf joints: (4, 2)."""
    co = contract.meta["tendon_coefs"]
    return np.array([[co[leg][f"{leg}_thigh"], co[leg][f"{leg}_calf"]] for leg in LEG_NAMES], float)
