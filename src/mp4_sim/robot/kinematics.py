"""Firmware leg kinematics for the MPX platform.

This is a faithful port of the maths in the ESP32 firmware
(``main/main.c``: ``fRIK`` / ``fLIK`` / ``rRIK`` / ``rLIK`` and the constants
``L1``, ``L2``, ``NEUTRAL_Z``). Keeping it here means the RL package can plan
foot targets and sanity-check the MuJoCo model without a robot.

Geometry, seen from the side of one leg:

* ``L1`` thigh length, ``L2`` shin length, both mm.
* ``th1`` thigh angle from vertical, positive *forward* (+x).
* ``th2`` knee angle such that the shin sits at ``90 - th2`` from vertical
  (that is how the knee *servo* is wired: it sets the shin's angle, not the
  angle between thigh and shin — see ``docs``/``build_model.py``).
* ``x`` forward, ``z`` down, relative to the thigh pivot (mm).

Servo angles are the numbers ``servo_write()`` takes: degrees, ``0`` = the
stand pose, calibration already removed. Right legs (``FR``, ``RR``) negate
the thigh and knee signs; left legs (``FL``, ``RL``) do not.

No MuJoCo import here — this module is pure maths and always importable.
"""

from __future__ import annotations

import math

#: Thigh length, mm (main.c: L1).
L1 = 50.0
#: Shin length, mm (main.c: L2).
L2 = 56.0
#: Hip-pivot-to-foot height in the stand pose, mm (main.c: NEUTRAL_Z).
NEUTRAL_Z = 70.0

#: Servo id order, and the leg order used everywhere in this package.
LEG_ORDER = ("FR", "FL", "RR", "RL")
JOINT_SUFFIXES = ("hip", "thigh", "calf")
JOINT_ORDER = tuple(f"{leg}_{suffix}" for leg in LEG_ORDER for suffix in JOINT_SUFFIXES)
#: Servo ids (1-based, as the firmware numbers them) per leg: hip, thigh, knee.
SERVO_IDS = {
    "FR": (1, 2, 3),
    "FL": (4, 5, 6),
    "RR": (7, 8, 9),
    "RL": (10, 11, 12),
}
#: ``-1`` for the right legs, ``+1`` for the left ones (firmware IK sign).
SIDE = {"FR": -1, "FL": +1, "RR": -1, "RL": +1}

# ---- stand pose (servo angles all zero) as joint angles --------------------
TH1N_RAD = -math.acos((L1 * L1 + NEUTRAL_Z**2 - L2 * L2) / (2 * L1 * NEUTRAL_Z))
TH2N_RAD = math.asin((NEUTRAL_Z**2 - L1 * L1 - L2 * L2) / (2 * L1 * L2)) - TH1N_RAD
TH1N_DEG = math.degrees(TH1N_RAD)
TH2N_DEG = math.degrees(TH2N_RAD)
#: Thigh and shin angles from vertical in the stand pose (degrees, + forward).
STAND_THIGH_DEG = TH1N_DEG
STAND_SHIN_DEG = 90.0 - TH2N_DEG


def leg_ik(x_mm: float, z_mm: float, th0_deg: float = 0.0, side: int = -1) -> tuple[float, float, float]:
    """Firmware leg IK: foot ``(x, z)`` mm relative to the thigh pivot ->
    ``(hip, thigh, knee)`` servo angles in degrees.

    ``side`` is ``-1`` for the right legs, ``+1`` for the left ones (see
    :data:`SIDE`). Unreachable targets are returned anyway (clamped by the
    law of cosines' domain), which is what the firmware does too.
    """
    th0 = math.radians(th0_deg)
    zd = z_mm / math.cos(th0)
    length = math.hypot(x_mm, zd)
    phi = math.atan2(x_mm, zd)
    c1 = (L1 * L1 + length * length - L2 * L2) / (2 * L1 * length)
    c2 = (length * length - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    th1 = phi - math.acos(max(-1.0, min(1.0, c1)))
    th2 = math.asin(max(-1.0, min(1.0, c2))) - th1
    thigh = -math.degrees(th1 - TH1N_RAD)
    knee = math.degrees(th2 - TH2N_RAD)
    if side > 0:
        thigh, knee = -thigh, -knee
    return th0_deg, thigh, knee


def leg_fk(hip_deg: float, thigh_deg: float, knee_deg: float, side: int = -1) -> tuple[float, float]:
    """Inverse of :func:`leg_ik` (ignoring the hip, which does not move the
    foot in this plane): servo angles -> ``(x, z)`` mm, relative to the thigh
    pivot, x forward and z down."""
    # servo = ik_sign * (angle - angle_neutral): the thigh sign is `side` and
    # the knee sign is `-side` (right legs negate the thigh, left the knee).
    th1 = TH1N_RAD + side * math.radians(thigh_deg)   # thigh from vertical
    th2 = TH2N_RAD - side * math.radians(knee_deg)
    a1 = th1
    a2 = math.radians(90.0) - th2                     # shin from vertical
    x = L1 * math.sin(a1) + L2 * math.sin(a2)
    z = L1 * math.cos(a1) + L2 * math.cos(a2)
    return x, z


def leg_servo(leg: str, x_mm: float, z_mm: float, th0_deg: float = 0.0) -> tuple[float, float, float]:
    """Like :func:`leg_ik` but for a named leg, applying the firmware's signs."""
    return leg_ik(x_mm, z_mm, th0_deg, SIDE[leg])


def reach_limit_mm() -> float:
    """Furthest the foot can be from the thigh pivot, mm (= L1 + L2)."""
    return L1 + L2


def in_workspace(x_mm: float, z_mm: float, margin_mm: float = 1.0) -> bool:
    """Is a foot target reachable (a little inside the full extension)?"""
    return math.hypot(x_mm, z_mm) <= reach_limit_mm() - margin_mm
