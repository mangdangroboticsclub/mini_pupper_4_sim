"""The fitted servo sets, shared by the GPU arms (``arms.py``) and the CPU servo bank (``bank.py``). NumPy only.

A role is one of the three servos in an MD01 housing: ``abd`` (abduction, the ``*_hip`` joint), ``hip`` (the
``*_thigh`` joint) and ``knee`` (the ``*_knee_drive`` tendon). Every set is class md01vp, m6, at its class defaults
(kp 80, kd 400, kff 0.001, max_pwm 0.99, 12 V).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .. import paths

PHYSICS_DT = 0.001
ROLES = ("abd", "hip", "knee")
LEGS = ("FR", "FL", "RR", "RL")
# The MuJoCo target of each role, by name suffix: two joints and the knee's fixed tendon.
TARGET_SUFFIX = {"abd": "hip", "hip": "thigh", "knee": "knee_drive"}
# The twelve targets in servo order (servo ids 1..12: FR, FL, RR, RL x abduction, hip, knee).
SERVO_TARGETS = tuple(f"{leg}_{TARGET_SUFFIX[role]}" for leg in LEGS for role in ROLES)

# T5 ub06: the abduction and the knee run the md01/id_3 set, the hip md01/id_2.
T5UB06 = {"abd": paths.ACTUATOR_PARAMS / "md01" / "bam_t5ub06_id_3_m6.json",
          "hip": paths.ACTUATOR_PARAMS / "md01" / "bam_t5ub06_id_2_m6.json",
          "knee": paths.ACTUATOR_PARAMS / "md01" / "bam_t5ub06_id_3_m6.json"}


def role_of(target: str) -> str:
    """``"FR_knee_drive"`` -> ``"knee"``."""
    for role, suffix in TARGET_SUFFIX.items():
        if target.endswith("_" + suffix):
            return role
    raise ValueError(f"{target!r} is not a servo target (expected a name ending in "
                     f"{sorted('_' + s for s in TARGET_SUFFIX.values())})")


def command_delay_steps(params_path) -> int:
    """A set's fitted ``command_delay`` in physics steps: a command takes effect at the first 1 ms step at or after its
    time + delay, i.e. the delay rounded up. T5 ub06: id_3 0.02 ms -> 1, id_2 4.98 ms -> 5."""
    ms = float(json.loads(Path(params_path).read_text())["command_delay"]) * 1000.0
    return int(math.ceil(ms / (PHYSICS_DT * 1000.0) - 1e-9))


def load(params_path, scales: tuple = ()):
    """A fresh BAM model for one set, with ``scales`` ((parameter, factor), ...) applied as ``BamServo`` does."""
    from .bam.model import load_model

    model = load_model(str(params_path))
    for name, factor in scales:
        getattr(model, name).value *= float(factor)
    return model
