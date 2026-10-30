"""The T5 ub06 servo law the JSON fits in ``assets/actuators`` were identified with.

``md01vp`` is the AT32 in control mode 1: a position P term and a velocity D term
become a PWM duty, then a brushed DC plant. Friction is BAM m6, a budget. The
stopping clip opposes the torque that would halt the rotor in one step, and the
effort returned to the solver is the motor torque plus that friction.
"""

from __future__ import annotations

import math

import torch

# Firmware defaults of the robot's MD01 boards. The JSON scales the two gains.
_KP = 80.0
_KD = 400.0
_KFF = 0.001
_MAX_PWM = 0.99
_VIN = 12.0
# Duty per (rad/s) per unit kd, from a ramp probe on one MD01. The fit's kd_ratio scales it.
_KD_GAIN = 2.87e-5
_DEG_PER_RAD = 180.0 / math.pi


def delay_steps(seconds: float, dt: float) -> int:
    """A command delay in physics steps, rounded up."""
    if seconds <= 0.0 or dt <= 0.0:
        return 0
    return int(math.ceil(seconds / dt - 1e-9))


def _soft_deadzone(x: torch.Tensor, v0: float, knee: float) -> torch.Tensor:
    """``max(|x| - v0, 0)`` with a soft knee of sharpness ``knee``. A large knee is the hard drop."""
    a = x.abs() - v0
    ka = knee * a
    soft = torch.where(ka > 20.0, a, torch.log1p(torch.exp(ka.clamp(max=20.0))) / knee)
    return torch.sign(x) * soft.clamp(min=0.0)


class Md01vp:
    """Position to PWM, then ``tau = kt * i`` with ``i`` from ``duty * vin - kt_emf * dq`` through ``R`` and ``V0``."""

    def __init__(self):
        self.kt = 0.0
        self.kt_emf = 0.0
        self.R = 1.0
        self.V0 = 0.0
        self.knee_extra = 4.0
        self.error_gain_ratio = 1.0
        self.q_offset = 0.0
        self.command_delay = 0.0
        self.armature = 0.0
        self.kd_ratio = 1.0

    def set_params(self, data: dict) -> None:
        for name in ("kt", "kt_emf", "R", "V0", "knee_extra", "error_gain_ratio", "q_offset", "command_delay",
                     "armature", "kd_ratio"):
            if name in data:
                setattr(self, name, float(data[name]))

    def reset(self, env_ids=None) -> None:
        """The position loop has no state of its own. The command delay lives in the actuator."""

    def control(self, q_target, q, dq, dt):
        gain = _KP * _KFF * self.error_gain_ratio * _DEG_PER_RAD
        d = _KD * _KD_GAIN * self.kd_ratio * dq
        duty = (gain * (q_target - q) - d).clamp(-_MAX_PWM, _MAX_PWM)
        return duty * _VIN

    def torque(self, volts, torque_enable: bool, q, dq):
        if not torque_enable:
            volts = torch.zeros_like(dq)
        x = volts - self.kt_emf * dq
        current = _soft_deadzone(x, self.V0, self.knee_extra) / self.R
        return self.kt * current


def get_motor(name: str):
    if name != "md01vp":
        raise ValueError(f"this package runs the md01vp fits; got {name!r}")
    return Md01vp


class BamFrictionModel:
    """BAM m6: Coulomb, viscous, Stribeck, directional load, and a quadratic load term."""

    def __init__(self, data: dict):
        if data.get("model") != "m6":
            raise ValueError(f"expected friction model m6, got {data.get('model')!r}")
        self.p = {k: float(data[k]) for k in (
            "friction_base", "friction_stribeck", "friction_viscous", "dtheta_stribeck", "alpha",
            "load_friction_motor", "load_friction_external", "load_friction_motor_stribeck",
            "load_friction_external_stribeck", "load_friction_motor_quad", "load_friction_external_quad",
        )}

    @classmethod
    def from_params(cls, data: dict) -> "BamFrictionModel":
        return cls(data)

    def compute(self, motor_torque, external, velocity):
        p = self.p
        gearbox = (external * p["load_friction_external"] - motor_torque * p["load_friction_motor"]).abs()
        gearbox_s = (external * p["load_friction_external_stribeck"]
                     - motor_torque * p["load_friction_motor_stribeck"]).abs()
        stribeck = torch.exp(-((velocity / p["dtheta_stribeck"]).abs() ** p["alpha"]))
        opposite = torch.sign(external) != torch.sign(motor_torque)
        on_motor = (external.abs() < motor_torque.abs()).to(velocity.dtype)
        on_external = (external.abs() > motor_torque.abs()).to(velocity.dtype)
        quad = (on_motor * p["load_friction_external_quad"] * external.abs().square()
                + on_external * p["load_friction_motor_quad"] * motor_torque.abs().square())
        loss = (p["friction_base"] + gearbox + stribeck * p["friction_stribeck"]
                + gearbox_s * stribeck + stribeck * quad * opposite.to(velocity.dtype))
        return loss, torch.full_like(velocity, p["friction_viscous"])


def motor_plus_friction(friction_model, motor_torque, external, velocity, inertia, dt):
    """Motor torque plus the friction that opposes stopping the rotor in one step.

    The load sets the friction budget and the stopping torque. It is not added to the
    effort: the solver already applies gravity, Coriolis, and contact.
    """
    frictionloss, damping = friction_model.compute(motor_torque, external, velocity)
    tau_stop = motor_torque + external + (inertia / dt) * velocity
    budget = frictionloss + damping * velocity.abs()
    friction = -torch.sign(tau_stop) * torch.minimum(tau_stop.abs(), budget)
    return motor_torque + friction
