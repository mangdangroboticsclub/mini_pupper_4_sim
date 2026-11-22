"""The servo model's name and flags, without the simulator: what evaluation tools, the policy export and the CPU tests
need (``servo_model`` builds the actuators). See ``servo_model`` for the models.
"""
from __future__ import annotations

import math

NAME = "md300"              # the default model: the robot's servos
ACTUATORS = ("md01", "md300")

# What the robot's policy runner writes to every servo (control mode 1) and its frame's current cap; the runner's tests
# check these against its code.
ROBOT_KP, ROBOT_KD, ROBOT_KFF, ROBOT_MAX_PWM, ROBOT_CAP_MA = 80.0, 400.0, 0.001, 0.99, 2000
# The gain mapping for md300. Its servo firmware (the one the MD-300 model was fitted on, and the robot's) computes in
# control mode 1
#     duty = clip(pwm_scale * kp * (|err_deg| - 0.1) - D, +-max_pwm),  D = kd * 2.87e-5 * (pwm_scale / 0.001) * dq
# measured on the bench to 0.8 % at 12 V and 10 V and at pwm_scale 0.001 and 0.002, with a 0.1-0.19 deg dead band.
# md01 (BAM class md01vp) models control mode 1 with kff_current as the duty scale, no dead band and D not scaled. The
# robot's servo firmware scales by pwm_scale (parameter 12) instead and does not read kff_current in mode 1 (on the bench,
# units differing only in kff_current track alike). The frame's current cap clips the PID output before the scale, so it
# is a duty cap pwm_scale x cap. So pwm_scale = the runner's kff (0.001) keeps the commanded gains: the same duty per
# degree, the same D, and duty cap 0.001 x 2000 = 2.0 > max_pwm (only max_pwm binds). The bench units shipped with
# pwm_scale 0.001 in flash and the fits were taken at 0.001, kd 400, max_pwm 0.99, 12 V; the runner does not write
# pwm_scale, so the robot's servos run at whatever their flash holds (read it with `mp4link gains`).
MD300_FIRMWARE = {"kp": ROBOT_KP, "kd": ROBOT_KD, "pwm_scale": ROBOT_KFF, "max_pwm": ROBOT_MAX_PWM,
                  "cur_cap_ma": ROBOT_CAP_MA, "vin": 12.0}


def duty_per_ma(name: str = NAME) -> float:
    """The firmware's scale from the PID output (mA units) to duty in control mode 1: kff_current for md01, pwm_scale for
    md300. A frame cap of C mA is a duty cap of this x C."""
    check_name(name)
    return ROBOT_KFF if name == "md01" else float(MD300_FIRMWARE["pwm_scale"])


def check_name(name: str) -> str:
    if name not in ACTUATORS:
        raise ValueError(f"unknown actuator model {name!r} (known: {', '.join(ACTUATORS)})")
    return name


def _range(r):
    if r is None:
        return None
    lo, hi = (float(v) for v in r)
    if not lo <= hi:
        raise ValueError(f"range {r!r}: low > high")
    return (lo, hi)


def add_args(ap, evaluation: bool = False) -> None:
    """``--actuator`` and the randomisation flags (``options_from_args``). ``evaluation``: the default model is the
    policy's (its meta's ``actuator``, else md300)."""
    ap.add_argument("--actuator", choices=ACTUATORS, default=None if evaluation else NAME,
                    help="the servo model" + (" (default: the policy's meta, else md300)" if evaluation else
                                              " (default %(default)s)"))
    ap.add_argument("--supply-v", type=float, nargs=2, metavar=("LO", "HI"),
                    help="per-env servo supply voltage, uniform (V; a 3S LiPo: 10.5 12.6; default: the model's 12 V)")
    ap.add_argument("--supply-drop-ohm", type=float, nargs=2, metavar=("LO", "HI"),
                    help="per-env pack + wiring resistance (Ohm): the bus sags with the servos' total draw (default: none)")
    ap.add_argument("--r-scale", type=float, nargs=2, metavar=("LO", "HI"),
                    help="per-env winding resistance scale, uniform (heat: 1.0 1.4; default: none)")


def options_from_args(args) -> dict:
    """Options fields (``actuator``, ``supply_v_range``, ...) from ``add_args``'s flags (an unset ``--actuator`` is left out)."""
    out = {"supply_v_range": _range(getattr(args, "supply_v", None)),
           "supply_drop_ohm_range": _range(getattr(args, "supply_drop_ohm", None)),
           "r_scale_range": _range(getattr(args, "r_scale", None))}
    if getattr(args, "actuator", None) is not None:
        out["actuator"] = check_name(args.actuator)
    return out


def apply_md300_firmware(act, fw: dict) -> None:
    """Set an ``MD300Actuator``'s firmware configuration as its ``load_log`` does from a log's ``at32_*`` keys: the P
    gain's duty scale ``pwm_scale * 180/pi``, the D term scaled by ``pwm_scale / 0.001``, the duty ceiling
    ``min(max_pwm, pwm_scale * cur_cap_ma)`` (the frame's current cap clips the loop output before the duty scale)."""
    scale = float(fw["pwm_scale"])
    act.kp = float(fw["kp"])
    act.kd = float(fw["kd"])
    act.vin = float(fw["vin"])
    act.error_gain = scale * (180.0 / math.pi)
    act.kd_scale = scale / act.DEFAULT_PWM_SCALE
    cap = fw.get("cur_cap_ma")
    act.max_pwm = float(fw["max_pwm"]) if cap is None else min(float(fw["max_pwm"]), float(cap) * scale)
