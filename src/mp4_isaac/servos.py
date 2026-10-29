"""The twelve MD01 servos as one explicit Isaac Lab actuator.

Abduction and hip each drive one hinge. The knee is the MuJoCo tendon
coordinate ``c_thigh * q_thigh + c_calf * q_calf``: ``md01vp`` and m6 run in
that coordinate, and the torque splits back onto the thigh and the calf by
the same coefficients. Those two joints therefore cannot sit in separate
actuator groups, so this one group owns all twelve hinges.

The position the firmware sees is that plant coordinate (joint radians, or
the tendon length), which is what the policies were trained against. The
fits are ``bam_t5ub06`` id 2 on the hip and id 3 on abduction and the knee.
"""

from __future__ import annotations

import json
from dataclasses import field

import torch
from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from bam_actuators.actuators.bam_actuator import delay_steps, motor_plus_friction
from bam_actuators.friction import BamFrictionModel
from bam_actuators.motors import get_motor

from .paths import PHYSICS_DT, ROLES, contract_json, fit_json

LEG_ORDER = ("FR", "FL", "RR", "RL")
PREFIX = {"FR": "front_right", "FL": "front_left", "RR": "back_right", "RL": "back_left"}
# SolidWorks spelled the SKU1 rear-right thigh joint this way. The URDF keeps it.
JOINT_ALIASES = {"back_right_leg_upper_joint": "back_riht_leg_upper_joint"}


def reflected_armature() -> dict[str, float]:
    """Solver armature, the diagonal of the servo inertias MuJoCo puts on each hinge.

    Abduction carries the abduction rotor. The thigh carries the hip rotor plus
    the knee tendon's share, and the calf carries the rest of that tendon. The
    stopping clip keeps its own copy of the servo armature; that copy only sizes
    stiction, it is not the hinge inertia the solver integrates.
    """
    values = {role: float(json.loads(fit_json(role).read_text())["armature"]) for role in ROLES}
    thigh = values["hip"] + values["knee"]
    return {
        ".*_hip_joint": values["abd"],
        ".*leg_upper_joint": thigh,
        ".*_leg_lower_joint": values["knee"],
    }


def _hinge(leg: str, role: str) -> str:
    prefix = PREFIX[leg]
    suffix = {"abd": "hip_joint", "hip": "leg_upper_joint", "knee": "leg_lower_joint"}[role]
    return f"{prefix}_{suffix}"


class PupperBamActuator(ActuatorBase):
    """Explicit m6 actuator for one Mini Pupper."""

    cfg: PupperBamActuatorCfg
    is_implicit_model = False

    def __init__(self, cfg: PupperBamActuatorCfg, *args, stiffness: float = 0.0, damping: float = 0.0, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.stiffness = stiffness
        self.damping = damping
        self._dt = float(cfg.physics_dt)
        names = list(self._joint_names)
        coefs = json.loads(contract_json().read_text())["tendon_coefs"]

        def column(name: str) -> int:
            if name in names:
                return names.index(name)
            alias = JOINT_ALIASES.get(name)
            if alias is not None and alias in names:
                return names.index(alias)
            raise KeyError(f"{name} is not in this actuator's joints: {names}")

        self._abd = [column(_hinge(leg, "abd")) for leg in LEG_ORDER]
        self._thigh = [column(_hinge(leg, "hip")) for leg in LEG_ORDER]
        self._calf = [column(_hinge(leg, "knee")) for leg in LEG_ORDER]
        self._c_thigh = torch.tensor(
            [float(coefs[leg][f"{leg}_thigh"]) for leg in LEG_ORDER], dtype=torch.float, device=self._device
        )
        self._c_calf = torch.tensor(
            [float(coefs[leg][f"{leg}_calf"]) for leg in LEG_ORDER], dtype=torch.float, device=self._device
        )

        self._motor = {}
        self._friction = {}
        self._q_offset = torch.zeros(12, device=self._device)
        self._armature = torch.zeros(12, device=self._device)
        self._lag = [0] * 12
        for slot, role in enumerate(ROLES):
            path = fit_json(role)
            data = json.loads(path.read_text())
            motor = get_motor(str(data["actuator"]))()
            motor.set_params(data)
            self._motor[role] = motor
            self._friction[role] = BamFrictionModel.from_params(data)
            lag = delay_steps(float(motor.command_delay), self._dt)
            for leg in range(4):
                index = leg * 3 + slot
                self._q_offset[index] = float(motor.q_offset)
                self._armature[index] = float(motor.armature)
                self._lag[index] = lag
            print(
                f"[PupperBamActuator] {role}: {path.name}  armature={float(motor.armature):.6g}  "
                f"command_delay={lag} steps"
            )
        self._max_lag = max(self._lag)
        self._cmd_hist: list[torch.Tensor] = []
        self._prev_vel = torch.zeros(self._num_envs, 12, device=self._device)
        self._prev_motor = torch.zeros(self._num_envs, 12, device=self._device)
        self._have_prev = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        self.servo_motor_torque = torch.zeros(self._num_envs, 12, device=self._device)

    def reset(self, env_ids=None) -> None:
        self._cmd_hist.clear()
        if env_ids is None or env_ids == slice(None):
            self._prev_vel.zero_()
            self._prev_motor.zero_()
            self._have_prev.fill_(False)
            ids = ...
        else:
            self._prev_vel[env_ids] = 0.0
            self._prev_motor[env_ids] = 0.0
            self._have_prev[env_ids] = False
            ids = env_ids
        for motor in self._motor.values():
            motor.reset(ids)

    def _plant(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Twelve servo coordinates: hip joint, thigh joint, knee tendon."""
        q = torch.empty(joint_pos.shape[0], 12, device=joint_pos.device, dtype=joint_pos.dtype)
        dq = torch.empty_like(q)
        q[:, 0::3] = joint_pos[:, self._abd]
        q[:, 1::3] = joint_pos[:, self._thigh]
        q[:, 2::3] = self._c_thigh * joint_pos[:, self._thigh] + self._c_calf * joint_pos[:, self._calf]
        dq[:, 0::3] = joint_vel[:, self._abd]
        dq[:, 1::3] = joint_vel[:, self._thigh]
        dq[:, 2::3] = self._c_thigh * joint_vel[:, self._thigh] + self._c_calf * joint_vel[:, self._calf]
        return q, dq

    def _delayed(self, target: torch.Tensor) -> torch.Tensor:
        if self._max_lag <= 0:
            return target
        self._cmd_hist.append(target.detach().clone())
        overflow = len(self._cmd_hist) - (self._max_lag + 1)
        if overflow > 0:
            del self._cmd_hist[:overflow]
        held = len(self._cmd_hist)
        out = torch.empty_like(target)
        for index, lag in enumerate(self._lag):
            slot = 0 if held <= lag else held - (lag + 1)
            out[:, index] = self._cmd_hist[slot][:, index]
        return out

    def _to_joints(self, tau: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        effort = torch.zeros_like(like)
        effort[:, self._abd] = tau[:, 0::3]
        effort[:, self._thigh] = tau[:, 1::3] + self._c_thigh * tau[:, 2::3]
        effort[:, self._calf] = self._c_calf * tau[:, 2::3]
        return effort

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        q, dq = self._plant(joint_pos, joint_vel)
        if control_action.joint_positions is None:
            target = q
        else:
            target, _ = self._plant(control_action.joint_positions, torch.zeros_like(joint_vel))
        target = self._delayed(target)

        external = torch.zeros_like(q)
        if bool(self._have_prev.any()):
            estimated = self._armature * (dq - self._prev_vel) / self._dt - self._prev_motor
            external = torch.where(self._have_prev[:, None], estimated, external)

        motor_torque = torch.empty_like(q)
        applied = torch.empty_like(q)
        for slot, role in enumerate(ROLES):
            sl = slice(slot, None, 3)
            motor = self._motor[role]
            q_r = q[:, sl]
            dq_r = dq[:, sl]
            control = motor.control(target[:, sl], q_r, dq_r, self._dt)
            torque = motor.torque(control, True, q_r + self._q_offset[sl], dq_r)
            motor_torque[:, sl] = torque
            applied[:, sl] = motor_plus_friction(
                self._friction[role],
                torque,
                external[:, sl],
                dq_r,
                self._armature[sl],
                self._dt,
            )

        self._prev_vel = dq.detach()
        self._prev_motor = motor_torque.detach()
        self._have_prev.fill_(True)
        self.servo_motor_torque = motor_torque
        self.computed_effort = self._to_joints(motor_torque, joint_pos)
        self.applied_effort = self._clip_effort(self._to_joints(applied, joint_pos))
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class PupperBamActuatorCfg(ActuatorBaseCfg):
    """Twelve hinges, three BAM roles. Solver armature and friction stay zero."""

    class_type: type = PupperBamActuator
    joint_names_expr: list[str] = [".*"]
    physics_dt: float = PHYSICS_DT
    stiffness: float = 0.0
    damping: float = 0.0
    armature: dict = field(default_factory=reflected_armature)
    friction: float = 0.0
    dynamic_friction: float = 0.0
    viscous_friction: float = 0.0
    joint_effort_limit: float = 3.0
    actuator_effort_limit: float = 3.0
