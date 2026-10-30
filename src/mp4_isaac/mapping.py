"""Firmware servo angles (degrees, 0 = the stand) to the CAD hinge angles, and back.

The map is affine, so affine() builds q = A @ servo_deg + b from servo_to_joint. Order is
FR, FL, RR, RL, and within a leg hip, thigh, calf.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .paths import servo_map_json

LEG_ORDER = ("FR", "FL", "RR", "RL")


class ServoMap:
    def __init__(self, path: str | Path | None = None):
        path = Path(path) if path is not None else servo_map_json()
        with open(path) as fh:
            self.cfg = json.load(fh)
        fw = self.cfg["firmware"]
        self.path = path
        self.th1n_deg = float(fw["th1n_deg"])
        self.th2n_deg = float(fw["th2n_deg"])
        self.legs = self.cfg["legs"]

    def stand_joint_angles(self) -> np.ndarray:
        """Hinge angles (rad, 12) when every servo is at 0."""
        return self.servo_to_joint(np.zeros(12))

    def servo_to_joint(self, servo_deg) -> np.ndarray:
        servo = np.asarray(servo_deg, dtype=float).reshape(12)
        q = np.zeros(12)
        for k, leg in enumerate(LEG_ORDER):
            leg_cfg = self.legs[leg]
            ids = leg_cfg["servos"]
            axis = leg_cfg["axis_sign"]
            zero = leg_cfg["zero_pose_deg"]
            a_hip = servo[ids["hip"] - 1]
            a_thigh = servo[ids["thigh"] - 1]
            a_knee = servo[ids["knee"] - 1]
            th1 = self.th1n_deg + leg_cfg["ik_sign"]["thigh"] * a_thigh
            th2 = self.th2n_deg + leg_cfg["ik_sign"]["knee"] * a_knee
            a1 = th1
            a2 = 90.0 - th2
            q[3 * k + 0] = math.radians(axis["hip"] * leg_cfg["hip_dir"] * a_hip)
            q[3 * k + 1] = math.radians(-axis["thigh"] * (a1 - zero["thigh"]))
            q[3 * k + 2] = math.radians(-axis["knee"] * ((a2 - a1) - (zero["shin"] - zero["thigh"])))
        return q

    def joint_to_servo(self, joint_rad) -> np.ndarray:
        q = np.asarray(joint_rad, dtype=float).reshape(12)
        servo = np.zeros(12)
        for k, leg in enumerate(LEG_ORDER):
            leg_cfg = self.legs[leg]
            ids = leg_cfg["servos"]
            axis = leg_cfg["axis_sign"]
            zero = leg_cfg["zero_pose_deg"]
            ik = leg_cfg["ik_sign"]
            q_hip, q_thigh, q_knee = (math.degrees(q[3 * k + i]) for i in range(3))
            a_hip = q_hip / (axis["hip"] * leg_cfg["hip_dir"])
            a1 = zero["thigh"] - q_thigh / axis["thigh"]
            a2 = a1 + (zero["shin"] - zero["thigh"]) - q_knee / axis["knee"]
            servo[ids["hip"] - 1] = a_hip
            servo[ids["thigh"] - 1] = (a1 - self.th1n_deg) / ik["thigh"]
            servo[ids["knee"] - 1] = ((90.0 - a2) - self.th2n_deg) / ik["knee"]
        return servo

    def affine(self):
        """(A, b) with q = A @ servo_deg + b, built from servo_to_joint."""
        if getattr(self, "_affine", None) is None:
            b = self.servo_to_joint(np.zeros(12))
            a = np.column_stack([self.servo_to_joint(np.eye(12)[:, j]) - b for j in range(12)])
            self._affine = (a, b)
        return self._affine
