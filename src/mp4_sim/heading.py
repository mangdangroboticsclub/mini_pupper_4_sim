"""The outer heading loop's control law, shared by the robot-side program (on the Mini Pupper's Raspberry Pi; numpy
floats) and the evaluation tools (torch tensors, one value per env): PI on the heading error into a yaw-rate command,
clipped, with the integrator clamped and, by default, conditional integration (the integrator holds while the output is
saturated in the error's direction). numpy only at import: the robot has no torch."""
from __future__ import annotations

import numpy as np

KP, KI, WZ_MAX, I_MAX = 2.0, 0.5, 0.4, 0.5
ANTIWINDUP = ("cond", "none")


def _ops(x):
    if hasattr(x, "clamp"):                                   # a torch tensor
        import torch
        return (lambda v, lo, hi: v.clamp(lo, hi)), torch.where
    return np.clip, np.where


def pi_step(err, integ, dt: float, kp: float = KP, ki: float = KI, wz_max: float = WZ_MAX, i_max: float = I_MAX,
            antiwindup: str = "cond"):
    """One step: ``err`` the heading error (rad, + = turn left to correct), ``integ`` the integrator (rad s).
    Returns (wz, new integ). ``antiwindup``: "none" integrates always (only the clamp at ±i_max); "cond" skips the
    update when the output before it is at the limit and the error has the output's sign (standard clamping: while
    saturated, an integrator of the opposite sign is held too, so it unwinds once |kp·err + ki·integ| < wz_max)."""
    if antiwindup not in ANTIWINDUP:
        raise ValueError(f"antiwindup must be one of {ANTIWINDUP}")
    clip, where = _ops(err)
    nxt = clip(integ + err * dt, -i_max, i_max)
    if antiwindup == "cond":
        u0 = kp * err + ki * integ
        nxt = where((abs(u0) >= wz_max) & (err * u0 > 0), integ, nxt)
    return clip(kp * err + ki * nxt, -wz_max, wz_max), nxt


LAT_MAX = 0.3                   # rad: the largest heading offset the lateral loop asks for


def lateral_step(y_hat, psi, v: float, dt: float, k: float, max_rad: float = LAT_MAX):
    """A dead-reckoned lateral loop around the heading loop: ``y_hat`` (m, + = left of the start line) integrates the
    commanded speed ``v`` along the heading ``psi`` (rad, relative to the start, + = left); the heading target becomes
    ``clip(-k · y_hat, ±max_rad)`` (rad) instead of the start heading. It sees only drift that comes from the heading
    (yaw from a disturbance), not sideways drift at a constant heading. Returns (target offset, new y_hat)."""
    clip, _ = _ops(psi)
    sin = psi.sin() if hasattr(psi, "sin") else np.sin(psi)
    y = y_hat + v * sin * dt
    return clip(-k * y, -max_rad, max_rad), y
