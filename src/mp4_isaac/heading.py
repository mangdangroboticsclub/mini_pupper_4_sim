"""Heading loop: PI on the heading error into a yaw-rate command.

The integrator is clamped. With antiwindup "cond" it holds while the output is saturated in the error's
direction. Works on a numpy float or a torch tensor.
"""

from __future__ import annotations

import numpy as np

KP, KI, WZ_MAX, I_MAX = 2.0, 0.5, 0.4, 0.5
ANTIWINDUP = ("cond", "none")


def _ops(x):
    if hasattr(x, "clamp"):
        import torch
        return (lambda v, lo, hi: v.clamp(lo, hi)), torch.where
    return np.clip, np.where


def pi_step(err, integ, dt, kp=KP, ki=KI, wz_max=WZ_MAX, i_max=I_MAX, antiwindup="cond"):
    """err in rad (+ = turn left). Returns (wz, new integrator)."""
    if antiwindup not in ANTIWINDUP:
        raise ValueError(f"antiwindup must be one of {ANTIWINDUP}")
    clip, where = _ops(err)
    nxt = clip(integ + err * dt, -i_max, i_max)
    if antiwindup == "cond":
        u0 = kp * err + ki * integ
        nxt = where((abs(u0) >= wz_max) & (err * u0 > 0), integ, nxt)
    return clip(kp * err + ki * nxt, -wz_max, wz_max), nxt
