"""The sensor noise measured on a robot (``robot/assets/<sku>/measured/noise_measured.json``): per channel and element,
the half-width of a uniform noise that reproduces the robot's logs (the sensor noise plus the broadband excess of the
logs over the simulation in motion). How a task applies it (and ramps it) is the task's."""
from __future__ import annotations

import json

from mp4_sim import robots


def noise_profile(name: str = "measured", robot=None) -> dict:
    """The noise's end amplitudes per channel (``ang_vel``, ``gravity``, ``pos``, ``vel``) and element, as measured on
    the robot (``robots.Robot.noise_json``; the SKU1 uses the SKU2's)."""
    if name != "measured":
        raise ValueError(f"unknown noise profile {name!r}")
    prof = json.loads(robots.get(robot).noise_json.read_text())["profile_uniform_half_width"]
    return {ch: [float(v) for v in vals] for ch, vals in prof.items()}
