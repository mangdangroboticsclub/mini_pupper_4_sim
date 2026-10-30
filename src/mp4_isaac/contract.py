"""What the network sees and what its output does, in numpy.

Per servo (FR, FL, RR, RL x abduction, hip, knee) the policy works in the servo's own angle in radians, relative
to the stand: rel = sign * radians(servo_deg). servo_deg is the angle the firmware's servo_write takes without the
calibration offset (0 = the stand). Signs and offsets come from assets/sku_2/sku2_contract.json.

Firmware units: tick = 511 + trunc((servo_deg + offset) * 10240 / 3100), dd = trunc(3100 - tick * 3100 / 1024).
A measured angle comes back as servo_deg = (DD0 - dd) / 10 - offset.

Observation, 45 floats: angular velocity (3), gravity (3, -acc/|acc|), command (3), position (12), velocity (12,
median of 3 then a 4-sample slope), previous applied action (12). A history of h steps is each term's frames,
oldest to newest, concatenated; after a reset every slot holds the first frame.

Action: clip to [-1, 1], scale * a, the band about the stand (20 deg abductions, 60 deg hips and knees), at most
20 deg from the previous applied target, then the firmware tick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .paths import contract_json

DD0 = 3100.0 - 511.0 * 3100.0 / 1024.0
STATE_DT = 0.005
CONTROL_DT = 0.020
STEP_LIMIT_DEG = 20.0
ABDUCTION = (0, 3, 6, 9)
BAND_DEG = np.array([20.0 if i in ABDUCTION else 60.0 for i in range(12)])
OBS_SIZE = 45
ACTION_SIZE = 12
F32 = np.float32

OBS_TERMS = (("ang_vel", slice(0, 3)), ("gravity", slice(3, 6)), ("command", slice(6, 9)), ("pos", slice(9, 21)),
             ("vel", slice(21, 33)), ("prev_action", slice(33, 45)))


def obs_size(history: int = 1) -> int:
    """Input width with a history of history steps (0 or 1: the 45-value frame)."""
    return OBS_SIZE * max(int(history), 1)


@dataclass
class Contract:
    signs: np.ndarray
    stand_rel_rad: np.ndarray
    stand_joint_rad: np.ndarray
    offsets_deg: np.ndarray
    stand_dd: np.ndarray
    stand_ticks: np.ndarray
    joint_order: list
    policy_targets: list
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Contract":
        d = json.loads(Path(path or contract_json()).read_text())
        return cls(signs=np.array(d["signs"], dtype=float), stand_rel_rad=np.array(d["stand_policy_rad"]),
                   stand_joint_rad=np.array(d["stand_joint_rad"]), offsets_deg=np.array(d["offsets_deg"]),
                   stand_dd=np.array(d["stand_dd"]), stand_ticks=np.array(d["stand_ticks"]),
                   joint_order=d["joint_order"], policy_targets=d["policy_targets"], meta=d)

    def ticks_from_servo_deg(self, servo_deg) -> np.ndarray:
        ang = (np.asarray(servo_deg, dtype=F32) + self.offsets_deg.astype(F32)).astype(F32)
        sig = 511 + np.trunc(ang * F32(10240.0) / F32(3100.0)).astype(np.int64)
        return np.clip(sig, 0, 1023)

    @staticmethod
    def dd_from_ticks(ticks) -> np.ndarray:
        t = np.asarray(ticks, dtype=F32)
        return np.trunc(F32(3100.0) - t * F32(3100.0) / F32(1024.0)).astype(np.int64)

    def servo_deg_from_dd(self, dd) -> np.ndarray:
        return (DD0 - np.asarray(dd, dtype=float)) / 10.0 - self.offsets_deg

    def rel_from_servo_deg(self, servo_deg) -> np.ndarray:
        return self.signs * np.radians(np.asarray(servo_deg, dtype=float))

    def servo_deg_from_rel(self, rel) -> np.ndarray:
        return np.degrees(np.asarray(rel, dtype=float)) * self.signs

    def rel_from_dd(self, dd) -> np.ndarray:
        return self.rel_from_servo_deg(self.servo_deg_from_dd(dd))

    def quantise_rel(self, rel):
        """The dd the firmware would send, and that dd back in policy coordinates."""
        dd = self.dd_from_ticks(self.ticks_from_servo_deg(self.servo_deg_from_rel(rel)))
        return dd, self.rel_from_dd(dd)


