"""The fitted servo model: what turns a policy target into a joint torque.

Two fits of MangDang's MD-300-150 servo module (the firmware calls it MD01), selected by name
(``SceneOptions.actuator``):

- ``md300`` (the default, the robot's servos): BAM's ``MD300Actuator`` (Rhoban/bam pull request #26, MangDang
  MD-300-150), model m6, one parameter set per output: id_1 abduction, id_2 hip (thigh), id_3 knee
  (``actuators/params/md300/md300_id_{1,2,3}_m6.json``, byte copies). Voltage-mode firmware law with a 0.1 deg dead band,
  ``I = (V - kt_emf * dq) / R`` (no knee). Gains: ``MD300_FIRMWARE``.
- ``md01``, an earlier fit: BAM (``actuators/bam``, class md01vp, model m6), refitted on bench data with the armature pinned
  at 0.6 of a bound measured on the bare output arm (knee 2.2e-4, hip 2.7e-4 kg m^2) and kd_ratio at 1.0, which predicts
  the robot's joints better than the published parameters at its light loads. The knee and abduction servos use one
  parameter file, the hip another (``actuators/params/md01/bam_t5ub06_id_{3,2}_m6.json``). At kp 80, kd 400, kff 0.001, max_pwm 0.99, 12 V; a soft V0 conduction knee.

Each set's fitted command delay is applied on the 1 ms physics grid, rounded up. The sensing-to-command delay, the same
for every servo, is added at reset by the task's command-delay event; every actuator holds its lag for the whole episode
(``delay_hold_prob = 1``: mjlab otherwise resamples a lag at every physics step, and actuator groups with different
transmissions keep separate buffers).

Per-env randomisation of the servos' supply and windings (``build``; off unless a range is given; drawn once at startup,
shared by the robot's twelve servos, see ``bam_servo``): the supply voltage (a 3S LiPo spans 10.5-12.6 V), the pack and
wiring resistance (the bus sags with the servos' total draw), and a scale on the winding resistance (heat: the bench
hip's effective R rose 8.9 -> 12.9 Ohm over a 30 s stall, x1.45).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from mjlab.actuator.actuator import TransmissionType

from mp4_sim import paths
from mp4_sim.actuators.bam_servo import BamServoCfg
from mp4_sim.actuators.choice import (  # noqa: F401
    ACTUATORS,
    MD300_FIRMWARE,
    NAME,
    ROBOT_CAP_MA,
    ROBOT_KD,
    ROBOT_KFF,
    ROBOT_KP,
    ROBOT_MAX_PWM,
    _range,
    add_args,
    check_name,
    duty_per_ma,
    options_from_args,
)

PHYSICS_DT = 0.001
MAX_LAG_STEPS = 60          # the delay buffers' capacity (ms at 1 ms physics); lags are set per episode below it
MD01_PARAMS = paths.ACTUATOR_PARAMS / "md01"
MD300_PARAMS = paths.ACTUATOR_PARAMS / "md300"
ROLES = (("abd", (r".*_hip",), MD01_PARAMS / "bam_t5ub06_id_3_m6.json", TransmissionType.JOINT),
         ("hip", (r".*_thigh",), MD01_PARAMS / "bam_t5ub06_id_2_m6.json", TransmissionType.JOINT),
         ("knee", (r".*_knee_drive",), MD01_PARAMS / "bam_t5ub06_id_3_m6.json", TransmissionType.TENDON))
MD300_ROLES = (("abd", (r".*_hip",), MD300_PARAMS / "md300_id_1_m6.json", TransmissionType.JOINT),
               ("hip", (r".*_thigh",), MD300_PARAMS / "md300_id_2_m6.json", TransmissionType.JOINT),
               ("knee", (r".*_knee_drive",), MD300_PARAMS / "md300_id_3_m6.json", TransmissionType.TENDON))


@dataclass(frozen=True)
class ServoModel:
    name: str
    actuators: tuple            # mjlab ActuatorCfg, in the entity's order
    latency_steps: tuple        # each actuator config's own command latency, physics steps
    params: dict                # every parameter, echoed into the run config


def command_delay_steps(params_path) -> int:
    """A parameter file's ``command_delay`` in physics steps: a command takes effect at the first 1 ms step at or after
    its time + delay, i.e. the delay rounded up."""
    ms = float(json.loads(Path(params_path).read_text())["command_delay"]) * 1000.0
    return int(math.ceil(ms / (PHYSICS_DT * 1000.0) - 1e-9))


def build(name: str = NAME, supply_v_range=None, supply_drop_ohm_range=None, r_scale_range=None) -> ServoModel:
    """The actuator set ``name`` (``ACTUATORS``) with the optional per-env randomisation: ``supply_v_range`` (V),
    ``supply_drop_ohm_range`` (Ohm, pack + wiring), ``r_scale_range`` (x the fitted winding resistance); None: off."""
    check_name(name)
    v, drop, rs = _range(supply_v_range), _range(supply_drop_ohm_range), _range(r_scale_range)
    if v is not None and v[0] <= 0:
        raise ValueError(f"supply_v_range {v}: must be > 0 V")
    if rs is not None and rs[0] <= 0:
        raise ValueError(f"r_scale_range {rs}: must be > 0")
    roles_def = ROLES if name == "md01" else MD300_ROLES
    firmware = None if name == "md01" else dict(MD300_FIRMWARE)
    cfgs, lat, roles = [], [], {}
    for role, targets, path, transmission in roles_def:
        delay = command_delay_steps(path)
        cfgs.append(BamServoCfg(target_names_expr=targets, transmission_type=transmission, vin_range=v,
                                vin_drop_resistance_range=drop, r_scale_range=rs, firmware=firmware,
                                stiff_frictionloss=False, delay_min_lag=0, delay_max_lag=MAX_LAG_STEPS,
                                delay_hold_prob=1.0, json_path=str(path)))
        lat.append(delay)
        roles[role] = {"params": paths.name(path), "sha256": paths.sha256(path),
                       "command_delay_steps": delay}
    params = {"roles": roles}
    if firmware is not None:
        params["firmware"] = firmware
    if v is not None or drop is not None or rs is not None:
        params["randomise"] = {"supply_v_range": v, "supply_drop_ohm_range": drop, "r_scale_range": rs}
    return ServoModel(name, tuple(cfgs), tuple(lat), params)


def md01() -> ServoModel:
    return build("md01")


def md300() -> ServoModel:
    return build("md300")


def for_options(opt) -> ServoModel:
    """The model an options object names (``actuator``, ``supply_v_range``, ``supply_drop_ohm_range``,
    ``r_scale_range``)."""
    return build(getattr(opt, "actuator", NAME), getattr(opt, "supply_v_range", None),
                 getattr(opt, "supply_drop_ohm_range", None), getattr(opt, "r_scale_range", None))
