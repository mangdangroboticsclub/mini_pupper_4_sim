"""A robot's measured joint limits in the model.

Source: the robot's limits file (SKU2: ``robot/assets/sku_2/measured/sku2_joint_limits.json``), its sha256 recorded in
``<stem>.sha256`` beside it (``load`` refuses a file that differs). Per leg, in physical terms (deg from the calibrated
stand): the abduction's range by hand (``abd_out_range``, + = outward) and under power (``abd_out_range_powered``), the
knee relative to the thigh (``knee_fold_range``, by hand), the potentiometers' track ends, the signs from the servos' dd
direction to the physical directions, the hand ranges as the model's joint ranges (``model_joint_range_rad``: ``*_hip``
= abduction, ``*_calf`` = knee - thigh), and optionally ``thigh_stop_deg_backward``: a measured stop on the thigh, deg
backward from the stand, at every abduction (absent or null: none).

- ``joint_ranges()``: the model's ``*_hip`` and ``*_calf`` ranges at the narrower of hand and powered on each side (the
  motor sometimes pressed further than the hand), and a ``*_thigh`` range for each leg with a thigh stop
  (``thigh_stops_deg``). A model joint is its stand (``stand_joint_rad``) + radians(the dd-direction value), the
  physical value times ``signs_dd_to_phys``.
- ``apply_limits(spec, margin_deg)``: those ranges on an ``MjSpec`` (the joints limited), each side narrowed by the
  margin.
- ``hip_envelope()``: the hip's chassis stop towards the body's middle as a function of the abduction (per leg,
  piecewise linear, model rad).
- ``track_rel()``: each servo's potentiometer track ends in servo-relative coordinates.

Every function takes the robot (``robots.get``: a name, a ``Robot`` or None for SKU2) and optionally an explicit file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mp4_sim import paths, robots
from mp4_sim.calibration import ACTION_SIZE, TRACK_DD, Contract

LEGS = ("FR", "FL", "RR", "RL")
THIGH_STOP_KEY = "thigh_stop_deg_backward"
TRACK_MARGIN_DEG = 20.0           # powered motion is kept this far inside a potentiometer's track ends
# The default robot's files (SKU2)
LIMITS_JSON = robots.SKU2.limits_json
LIMITS_SHA256_FILE = robots.SKU2.limits_sha256_file
HIP_ENVELOPE_JSON = robots.SKU2.hip_envelope_json


def sha256_file(limits_json) -> Path:
    """A limits file's recorded sha256: ``<stem>.sha256`` beside it."""
    return Path(limits_json).with_suffix(".sha256")


def recorded_sha256(robot=None, limits_json=None) -> str:
    """The recorded sha256 of the robot's limits file (or of ``limits_json``)."""
    p = robots.get(robot).limits_json if limits_json is None else Path(limits_json)
    return sha256_file(p).read_text().split()[0]


LIMITS_SHA256 = recorded_sha256()        # SKU2's


def load(path=None, robot=None) -> dict:
    """The limits file: the robot's (its hash checked against the recorded one), or ``path`` (unchecked)."""
    if path is not None:
        return json.loads(Path(path).read_text())
    p = robots.get(robot).limits_json
    have, want = paths.sha256(p), recorded_sha256(limits_json=p)
    if have != want:
        raise ValueError(f"{paths.name(p)}: sha256 {have} is not the recorded {want} ({sha256_file(p).name})")
    return json.loads(p.read_text())


def limits_contract_check(allow_mismatch: bool = False, j: dict | None = None, robot=None,
                          contract_json=None) -> dict:
    """The limits' calibration against the calibration in use: the limits file's ``contract_sha256_prefix`` (the
    calibration file it was traced at: its sha256, ``paths.sha256``, 12 hex) and the same prefix of ``contract_json``
    (None: the robot's). Raises ``ValueError`` on a mismatch unless ``allow_mismatch`` (limits traced at another
    calibration, used knowingly). Returns the record for a run's config."""
    r = robots.get(robot)
    j = load(robot=r) if j is None else j
    want = str(j.get("contract_sha256_prefix") or "")
    path = r.contract_json if contract_json is None else Path(contract_json)
    have = paths.sha256(path)
    rec = {"limits_contract_sha256_prefix": want or None, "contract_in_force": paths.name(path),
           "contract_in_force_sha256_prefix": have[:12], "match": bool(want) and have.startswith(want),
           "allow_limits_contract_mismatch": bool(allow_mismatch)}
    if not rec["match"] and not allow_mismatch:
        raise ValueError(f"the limits were traced at calibration {want or '(none recorded)'}, the calibration in use "
                         f"({paths.name(path)}) is {have[:12]}: use limits fitted at this calibration, or allow the "
                         f"mismatch")
    return rec


def legs(j: dict | None = None, robot=None) -> dict:
    j = load(robot=robot) if j is None else j
    return {leg["leg"]: leg for leg in j["legs"]}


def thigh_stops_deg(j: dict | None = None, robot=None) -> dict:
    """Per leg with one: the thigh's measured stop, deg backward from the stand (``thigh_stop_deg_backward``)."""
    return {leg: float(d[THIGH_STOP_KEY]) for leg, d in legs(j, robot).items() if d.get(THIGH_STOP_KEY) is not None}


def physical_ranges(j: dict | None = None, robot=None) -> dict:
    """Per leg (deg from the stand, physical directions): ``abd_out`` (inward stop < 0 < outward stop) at the narrower
    of hand and powered on each side, and ``knee_fold`` (hand)."""
    out = {}
    for leg, d in legs(j, robot).items():
        hand, pw = d["abd_out_range"], d.get("abd_out_range_powered") or d["abd_out_range"]
        out[leg] = {"abd_out": (max(hand[0], pw[0]), min(hand[1], pw[1])),
                    "knee_fold": tuple(float(v) for v in d["knee_fold_range"])}
    return out


def model_range(stand: float, sign: float, phys: tuple) -> tuple:
    """A physical range (deg from the stand) as a model range (rad, absolute): stand + radians(sign * value)."""
    a, b = stand + np.radians(sign * phys[0]), stand + np.radians(sign * phys[1])
    return (float(min(a, b)), float(max(a, b)))


def joint_ranges(j: dict | None = None, contract: Contract | None = None, hand_only: bool = False,
                 robot=None) -> dict:
    """The model's joint ranges (rad, absolute): ``{"FR_hip": (lo, hi), "FR_calf": (lo, hi), "FR_thigh": (lo, hi),
    ...}``. ``hand_only``: the hand ranges (what the file's ``model_joint_range_rad`` holds). ``contract``: the
    calibration (None: the robot's)."""
    r = robots.get(robot)
    j = load(robot=r) if j is None else j
    c = Contract.load(r.contract_json) if contract is None else contract
    stand = dict(zip(c.joint_order, c.stand_joint_rad))
    phys = physical_ranges(j)
    out = {}
    for leg, d in legs(j).items():
        s = d["signs_dd_to_phys"]
        abd = tuple(d["abd_out_range"]) if hand_only else phys[leg]["abd_out"]
        out[f"{leg}_hip"] = model_range(stand[f"{leg}_hip"], s["abd_out"], abd)
        out[f"{leg}_calf"] = model_range(stand[f"{leg}_calf"], s["knee_fold"], phys[leg]["knee_fold"])
    for leg, stop in thigh_stops_deg(j).items():     # the free side is left to the servo's track ends
        li = LEGS.index(leg)
        back = -float(c.signs[3 * li + 1]) * float(legs(j)[leg]["signs_dd_to_phys"]["hip_fwd"]) < 0   # model + = backward
        st = stand[f"{leg}_thigh"]
        rad = float(np.radians(stop))
        out[f"{leg}_thigh"] = (st - 2.0 * np.pi, st + rad) if back else (st - rad, st + 2.0 * np.pi)
    return out


def apply_limits(spec, margin_deg: float = 0.0, ranges: dict | None = None, robot=None) -> dict:
    """Limit the ``*_hip`` (abduction) and ``*_calf`` (knee - thigh) joints (and a stopped ``*_thigh``) of an
    ``MjSpec`` to ``ranges`` (default ``joint_ranges(robot=robot)``), each side narrowed by ``margin_deg``. Returns
    the ranges set."""
    import mujoco
    ranges = joint_ranges(robot=robot) if ranges is None else ranges
    m = np.radians(float(margin_deg))
    out = {}
    for name, (lo, hi) in ranges.items():
        lo, hi = lo + m, hi - m
        if not lo < hi:
            raise ValueError(f"{name}: the margin {margin_deg} deg leaves no range")
        jt = spec.joint(name)
        jt.range = [lo, hi]
        jt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        out[name] = (lo, hi)
    return out


def abd_out_sign(j: dict | None = None, robot=None) -> tuple:
    """Per leg the model / servo-rel sign of an outward abduction, from the file (= ``calibration.ABD_OUT_SIGN``)."""
    return tuple(float(legs(j, robot)[leg]["signs_dd_to_phys"]["abd_out"]) for leg in LEGS)


def track_rel(contract: Contract | None = None, margin_deg: float = TRACK_MARGIN_DEG, robot=None) -> tuple:
    """Per servo (servo order) the rel (rad) ``margin_deg`` inside the potentiometer's track ends (dd ``TRACK_DD``):
    (lo, hi). A negative margin is past the ends. ``contract``: the calibration (None: the robot's)."""
    c = Contract.load(robots.get(robot).contract_json) if contract is None else contract
    m = 10.0 * float(margin_deg)
    a = c.rel_from_dd(np.full(ACTION_SIZE, TRACK_DD[0] + m))
    b = c.rel_from_dd(np.full(ACTION_SIZE, TRACK_DD[1] - m))
    return np.minimum(a, b), np.maximum(a, b)


def hip_envelope_path(robot=None) -> Path:
    """The robot's hip envelope file (SKU2: ``sku2_hip_envelope.json``)."""
    return robots.get(robot).hip_envelope_json


def hip_envelope(contract: Contract | None = None, path=None, robot=None) -> dict | None:
    """The hip envelope (``{leg: {"abd": [...], "lo": [...], "hi": [...]}}``, model rad), or None without the file.
    ``contract`` is accepted for symmetry with the other functions and not used: the file is in model coordinates."""
    p = hip_envelope_path(robot) if path is None else Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())["envelope"]
