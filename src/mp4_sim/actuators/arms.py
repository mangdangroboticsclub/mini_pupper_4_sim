"""The actuator arms: what turns a policy target into a joint torque.

An arm is a set of mjlab actuator configs plus each config's own command latency. The sensing-to-command chain delay
(the same for every arm) is added at reset by ``mdp.reset_command_delay``; every actuator holds its lag for the whole
episode (``delay_hold_prob = 1``: mjlab otherwise resamples a lag at every physics step, and actuator groups with
different transmissions keep separate buffers).

``stock``: the ``sim`` servo profile (robot/assets/servo_profiles.json): a MuJoCo position
actuator kp 8 N m/rad, kv 0.05, +-0.8 N m; damping 0.0381 and armature 2e-4 on the actuated joint, or on the knee's
tendon; its ``latency_ms`` (5) as the command latency. The XML's frictionloss stays (0.003 on the abduction and hip
joints, 0.001 on the passive calf hinge, none on the tendon), and so do the calf hinge's own damping and armature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.actuator.actuator import TransmissionType

from .. import paths
from .fits import PHYSICS_DT, T5UB06, command_delay_steps

MAX_LAG_STEPS = 60          # the delay buffers' capacity (ms at 1 ms physics); lags are set per episode below it
JOINT_TARGETS = (r".*_hip", r".*_thigh")
KNEE_TARGETS = (r".*_knee_drive",)


@dataclass(frozen=True)
class Arm:
    name: str
    actuators: tuple            # mjlab ActuatorCfg, in the entity's order
    latency_steps: tuple        # each actuator config's own command latency, physics steps
    params: dict                # every parameter, echoed into the run config


def stock_profile() -> dict:
    return json.loads(paths.SERVO_PROFILES.read_text())["profiles"]["sim"]


def stock() -> Arm:
    p = stock_profile()
    kw = dict(stiffness=p["kp"], damping=p["kv"], effort_limit=p["umax"], armature=p["armature"],
              viscous_damping=p["damping"], delay_min_lag=0, delay_max_lag=MAX_LAG_STEPS, delay_hold_prob=1.0)
    joints = BuiltinPositionActuatorCfg(target_names_expr=JOINT_TARGETS, **kw)
    knees = BuiltinPositionActuatorCfg(target_names_expr=KNEE_TARGETS, transmission_type=TransmissionType.TENDON, **kw)
    lat = int(round(p["latency_ms"] / 1000.0 / PHYSICS_DT))
    return Arm("stock", (joints, knees), (lat, lat),
               {"profile": "servo_profiles.json 'sim'", **{k: p[k] for k in
                ("kp", "kv", "umax", "damping", "armature", "latency_ms")},
                "source_sha256": paths.sha256(paths.SERVO_PROFILES)})


# The BAM arms: per role, the targets, the transmission, and the fitted set. Class md01vp, m6, at its class defaults
# (kp 80, kd 400, kff 0.001, max_pwm 0.99, 12 V). Each set's fitted command delay is applied on the 1 ms physics grid,
# rounded up.
BAM_ROLES = (("abd", JOINT_TARGETS[:1], TransmissionType.JOINT),
             ("hip", JOINT_TARGETS[1:], TransmissionType.JOINT),
             ("knee", KNEE_TARGETS, TransmissionType.TENDON))


def bam(name: str, params_json: dict, scales: tuple = (), vin_range: tuple | None = None,
        stiff_frictionloss: bool = False) -> Arm:
    """``params_json``: role -> a fitted JSON;
    ``scales``: ((parameter, factor), ...) on every role (e.g. ``armature`` for a reduced-armature variant);
    ``stiff_frictionloss``: False keeps MuJoCo's default friction constraint."""
    from .bam_servo import BamServoCfg
    cfgs, lat, params = [], [], {"roles": {}, "scales": [list(s) for s in scales], "vin_range": vin_range,
                                  "stiff_frictionloss": stiff_frictionloss}
    for role, targets, trn in BAM_ROLES:
        path = params_json[role]
        delay = command_delay_steps(path)
        cfgs.append(BamServoCfg(target_names_expr=targets, transmission_type=trn, vin_range=vin_range,
                                stiff_frictionloss=stiff_frictionloss, param_scales=tuple(scales),
                                delay_min_lag=0, delay_max_lag=MAX_LAG_STEPS, delay_hold_prob=1.0,
                                json_path=str(path)))
        lat.append(delay)
        params["roles"][role] = {"params": paths.rel(path), "sha256": paths.sha256(path), "command_delay_ms": delay}
    return Arm(name, tuple(cfgs), tuple(lat), params)


def t5ub06() -> Arm:
    """T5 ub06: a refit of md01vp m6 on the bench data with the armature pinned at 0.6 of a bound measured on the bare
    output arm (the knee 2.2e-4, the hip 2.7e-4 kg m^2) and kd_ratio at 1.0; it predicts SKU2's joints better than the
    published sets at the robot's light loads (actuators/params/md01/bam_t5ub06_id_*_m6.json)."""
    return bam("bam_t5ub06", T5UB06)


ARMS = {"stock": stock, "bam_t5ub06": t5ub06}


def trained_arm(path) -> str | None:
    """The arm a run or an exported policy was trained in: its ``config.json`` (in the directory, or next to a
    checkpoint)."""
    p = Path(path)
    for cfg in (p / "config.json", p.parent / "config.json"):
        if cfg.is_file():
            return json.loads(cfg.read_text())["task"]["arm"]
    return None


def get(name: str) -> Arm:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {sorted(ARMS)}")
    return ARMS[name]()
