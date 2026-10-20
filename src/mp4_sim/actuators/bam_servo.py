"""A BAM actuator for mjlab that drives joints or fixed tendons, matching the CPU servo bank.

``bam.mjlab``'s ``BamActuator`` drives joints only. This subclass keeps its control law, torque equation and
friction budget (``compute_control``, ``compute_torque``, ``_compute_friction_budget``) and changes, for the SKU2 knee
(a ``<fixed>`` tendon over the thigh and calf) and for parity with the CPU servo bank (``BamServoBank.update``,
not yet in this package):

- **The servo's position and velocity** are ``actuator_length`` / ``actuator_velocity`` (the bank's read: the values
  of the previous step's forward pass), for joints and tendons alike.
- **The motor torque the load-dependent friction compares against** is each actuator's own ``actuator_force``. The
  per-joint ``qfrc_actuator`` that ``BamActuator`` uses would charge the knee tendon's force to the thigh.
- **The external load** is ``-qfrc_bias + qfrc_constraint - (MuJoCo's own dof friction)`` on the actuator's dofs,
  projected on its coordinate (``sum(coef * load) / sum(coef^2)``; coef = 1 for a joint, the tendon's coefficients
  for a knee), minus its own tendon friction: the bank's projection (an approximation of the rod's statics).
- **The friction budget** goes into ``dof_frictionloss`` / ``dof_damping`` for joints, and ``tendon_frictionloss`` /
  ``tendon_damping`` for tendons, per environment (``bam_servo_init`` expands all four fields). The rotor inertia goes
  on the joint or the tendon, and the tendon's XML damping is replaced.

Not in this class (as in the bank): the fitted ``command_delay`` (the arm's actuator delay carries it) and the idle
bridge's braking (a policy never disables torque).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.actuator.actuator import Actuator, TransmissionType
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.spec import create_motor_actuator

from .bam.actuator import TorchBackend
from .bam.mjlab import BamActuator, BamActuatorCfg

_FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
_FRICTION_TENDON = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)


@requires_model_fields("dof_frictionloss", "dof_damping", "tendon_frictionloss", "tendon_damping")
def bam_servo_init(env, env_ids=None) -> None:
    """Startup event: expand the friction fields BamServo writes per environment."""
    del env, env_ids


@dataclass(kw_only=True)
class BamServoCfg(BamActuatorCfg):
    param_scales: tuple = ()          # ((parameter name, factor), ...), e.g. (("armature", 0.35),)

    def build(self, entity, target_ids, target_names):
        return BamServo(self, entity, target_ids, target_names)


class BamServo(BamActuator):
    cfg: BamServoCfg

    def __init__(self, cfg, entity, target_ids, target_names):
        super().__init__(cfg, entity, target_ids, target_names)
        for name, factor in cfg.param_scales:
            getattr(self._bam_model, name).value *= float(factor)

    @property
    def is_tendon(self) -> bool:
        return self.cfg.transmission_type == TransmissionType.TENDON

    def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
        if not self.is_tendon:
            return super().edit_spec(spec, target_names)
        armature = float(self._bam_model.actuator.get_extra_inertia())
        for name in target_names:
            mjact = create_motor_actuator(spec, name, effort_limit=0.0, armature=armature, frictionloss=0.0,
                                          viscous_damping=0.0, transmission_type=TransmissionType.TENDON)
            self._disable_limits(mjact)
            self._mjs_actuators.append(mjact)
            tendon = spec.tendon(name)
            tendon.armature = armature
            tendon.damping = np.zeros((3, 1))
            tendon.frictionloss = 0.0
            if self.cfg.stiff_frictionloss:
                tendon.solref_friction = self._STIFF_SOLREF_FRICTION
                tendon.solimp_friction = self._STIFF_SOLIMP_FRICTION

    def initialize(self, mj_model, model, data, device) -> None:
        Actuator.initialize(self, mj_model, model, data, device)
        self._mjwarp_model, self._data, self._device = model, data, device
        dofs, coefs, tendons = [], [], []
        for tid in self._target_ids_list:
            if self.is_tendon:
                t = int(self.entity.indexing.tendon_ids[tid])
                adr, num = mj_model.tendon_adr[t], mj_model.tendon_num[t]
                dofs.append([int(mj_model.jnt_dofadr[mj_model.wrap_objid[k]]) for k in range(adr, adr + num)])
                coefs.append([float(mj_model.wrap_prm[k]) for k in range(adr, adr + num)])
                tendons.append(t)
            else:
                j = int(self.entity.indexing.joint_ids[tid])
                dofs.append([int(mj_model.jnt_dofadr[j])])
                coefs.append([1.0])
                tendons.append(-1)
        width = max(len(d) for d in dofs)

        def pad(rows, v):
            return [r + [v] * (width - len(r)) for r in rows]

        self._dofs = torch.tensor(pad(dofs, 0), dtype=torch.long, device=device)            # (J, W)
        self._coefs = torch.tensor(pad(coefs, 0.0), dtype=torch.float32, device=device)     # (J, W)
        self._norm = (self._coefs ** 2).sum(-1)                                              # (J,)
        self._tendons = torch.tensor(tendons, dtype=torch.long, device=device)
        self._dof_ids = self._dofs[:, 0]
        n = data.nworld
        self._num_envs = n
        self._friction_fields_checked = False
        self.kp_scale = torch.ones(n, 1, device=device)
        self.kd_scale = torch.ones(n, 1, device=device)
        self.default_kp_scale, self.default_kd_scale = self.kp_scale.clone(), self.kd_scale.clone()
        bam = self._bam_model
        act = bam.actuator
        act.backend = TorchBackend()
        bam.reset()
        self._base_kp = float(act.kp)
        self._dt = float(mj_model.opt.timestep)
        if self.cfg.vin_range is not None:
            self.vin_tensor = torch.empty(n, 1, device=device).uniform_(*self.cfg.vin_range)
        else:
            self.vin_tensor = torch.full((n, 1), float(act.vin), device=device)
        self.vin_drop_resistance = None
        self.friction_scale = None
        if self.cfg.friction_scale_range is not None:
            self.friction_scale = torch.empty(n, 1, device=device).uniform_(*self.cfg.friction_scale_range)
        self._last = {}

    def _efc_friction(self, kind: int, size: int) -> torch.Tensor:
        efc = self._data.efc
        typ, ids, force = (self._as_tensor(efc.type), self._as_tensor(efc.id), self._as_tensor(efc.force))
        nefc = self._as_tensor(self._data.nefc)
        valid = torch.arange(typ.shape[1], device=typ.device)[None] < nefc[:, None]
        sel = valid & (typ == kind)
        out = torch.zeros(typ.shape[0], size, dtype=force.dtype, device=force.device)
        out.scatter_add_(1, torch.where(sel, ids, torch.zeros_like(ids)).long(), torch.where(sel, force, 0.0 * force))
        return out

    def compute(self, cmd):
        bam = self._bam_model
        act = bam.actuator
        ctrl = self.global_ctrl_ids
        pos = self._as_tensor(self._data.actuator_length)[:, ctrl]
        vel = self._as_tensor(self._data.actuator_velocity)[:, ctrl]
        act.vin = self.vin_tensor
        act.kp = self._base_kp * self.kp_scale
        control = act.compute_control(cmd.position_target, pos, vel * self.kd_scale, self._dt)
        torque = act.compute_torque(control, True, pos, vel * self.kd_scale)

        bias = self._as_tensor(self._data.qfrc_bias)
        constraint = self._as_tensor(self._data.qfrc_constraint)
        nv = bias.shape[-1]
        f_dof = self._efc_friction(_FRICTION_DOF, nv)
        load = -bias + constraint - f_dof                                                    # (N, nv)
        external = (load[:, self._dofs] * self._coefs).sum(-1) / self._norm                 # (N, J)
        if self.is_tendon:
            ntendon = self._mjwarp_model.tendon_frictionloss.shape[-1]
            external = external - self._efc_friction(_FRICTION_TENDON, ntendon)[:, self._tendons]
        motor = self._as_tensor(self._data.actuator_force)[:, ctrl]
        if bam.stribeck:
            stribeck = torch.exp(-torch.pow(torch.abs(vel) / bam.dtheta_stribeck.value, bam.alpha.value))
        else:
            stribeck = torch.zeros_like(vel)
        budget = self._compute_friction_budget(motor, external, stribeck)
        viscous = bam.friction_viscous.value
        if self.is_tendon:
            self._mjwarp_model.tendon_frictionloss[:, self._tendons] = budget
            self._mjwarp_model.tendon_damping[:, self._tendons] = viscous
        else:
            self._write_frictions(budget, viscous)
        self._last = {"pos": pos, "vel": vel, "torque": torque, "motor": motor, "external": external,
                      "frictionloss": budget}
        return torque
