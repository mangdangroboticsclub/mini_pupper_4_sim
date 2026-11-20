"""Contact-model edits for a robot that falls and gets up: self-collision between the legs, and stiffer contacts on
every collision geom but the feet. Pure ``MjSpec`` edits, applied by wrapping the robot's ``spec_fn``
(``wrap_spec_fn``).

Robot geoms are contype 1 / conaffinity 2, so they never collide with each other unless bits are added; the terrain is
1 / 1 and sees none of the bits below.

- ``getup_collisions``: ``LEG_BIT`` on every thigh and calf geom (the feet included) both ways: leg <-> leg;
  ``CAN_BIT`` on the legs' contype and the hip cans' conaffinity: leg <-> hip can; ``BASE_BIT`` (as
  ``robot.model.leg_base_collisions``: the base's conaffinity) on the thighs' contype, each thigh capsule shortened at
  its hip end by ``thigh_short_m`` (the model's thighs overlap the base box at the stand). MuJoCo filters parent-child
  pairs only, so a calf against its own leg's hip can stays a pair. With the measured joint limits the thigh <-> base
  pair is left off: its inward contact comes at 2-7 deg against the robot's measured 14-22, and a decoder's hip
  envelope covers it.
- ``stiff_contacts``: contact priority 2 and ``CONTACT_SOLREF`` on every collision geom but the feet.
"""
from __future__ import annotations

import mujoco
import numpy as np

from mp4_sim.robot.model import LEGS

LEG_BIT, CAN_BIT, BASE_BIT = 8, 16, 4
CONTACT_SOLREF = (0.005, 1.0)       # (timeconst, dampratio) of the body's and legs' contacts (MuJoCo's default 0.02)
THIGH_SHORT_M = 0.008               # 6 mm clears every thigh at the stand (FR by 0.01 mm); 8 mm leaves 0.16 mm


def getup_collisions(spec: mujoco.MjSpec, thigh_base: bool = True, thigh_short_m: float = THIGH_SHORT_M) -> dict:
    """The self-collision pairs (module docstring) on ``spec``. Returns what was changed."""
    out = {"thigh_capsule_m": {}}
    for leg in LEGS:
        for part in ("thigh", "calf"):
            for g in spec.body(f"{leg}_{part}").geoms:
                if g.contype or g.conaffinity:
                    g.contype |= LEG_BIT | CAN_BIT
                    g.conaffinity |= LEG_BIT
        for g in spec.body(f"{leg}_hip").geoms:
            if g.contype or g.conaffinity:
                g.conaffinity |= CAN_BIT
        if thigh_base:
            for g in spec.body(f"{leg}_thigh").geoms:
                if g.name == f"{leg}_thigh_collision":
                    ft = np.array(g.fromto, dtype=float)
                    a, b = ft[:3], ft[3:]
                    length = float(np.linalg.norm(b - a))
                    assert length - thigh_short_m >= 0.030, "under 30 mm of thigh capsule left: drop thigh <-> base"
                    g.fromto = np.r_[a + (b - a) / length * thigh_short_m, b]
                    g.contype |= BASE_BIT
                    out["thigh_capsule_m"][leg] = length - thigh_short_m
    for g in spec.body("base").geoms:          # leg_base_collisions sets it already (self_collision)
        if g.contype or g.conaffinity:
            g.conaffinity |= BASE_BIT
    return out


def stiff_contacts(spec: mujoco.MjSpec, solref: tuple = CONTACT_SOLREF) -> int:
    """Every robot collision geom but the feet: contact priority 2 and ``solref``. MuJoCo takes a contact's parameters
    from the geom of higher priority (equal: mixed by solmix), so these govern the body's and the legs' contacts with
    the floor (priority 0), with each other and with a foot (priority 1); the feet keep their own (priority 1, the
    scene's foot solref) against the floor and each other. Returns the geoms changed."""
    k = 0
    for b in spec.bodies:
        for g in b.geoms:
            if (g.contype or g.conaffinity) and not g.name.endswith("_foot"):
                g.priority = 2
                g.solref = list(solref)
                k += 1
    return k


def wrap_spec_fn(prev, collisions: bool, thigh_base: bool, thigh_short_m: float = THIGH_SHORT_M, solref=None):
    """The robot's ``spec_fn`` followed by ``getup_collisions`` (``collisions``) and ``stiff_contacts`` (``solref``
    not None)."""
    def fn():
        spec = prev()
        if collisions:
            getup_collisions(spec, thigh_base, thigh_short_m)
        if solref is not None:
            stiff_contacts(spec, solref)
        return spec
    return fn
