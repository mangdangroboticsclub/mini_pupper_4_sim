"""A BAM actuator for mjlab that drives joints or fixed tendons.

``bam/mjlab.py``'s ``BamActuator`` drives joints only. This subclass keeps its control law, torque equation and friction
budget (``compute_control``, ``compute_torque``, ``_compute_friction_budget``) and changes, for the knee (a ``<fixed>``
tendon over the thigh and calf):

- **The servo's position and velocity** are ``actuator_length`` / ``actuator_velocity`` (the values of the previous step's
  forward pass), for joints and tendons alike.
- **The motor torque the load-dependent friction compares against** is each actuator's own ``actuator_force``. The
  per-joint ``qfrc_actuator`` that ``BamActuator`` uses would charge the knee tendon's force to the thigh.
- **The external load** is ``-qfrc_bias + qfrc_constraint - (MuJoCo's own dof friction)`` on the actuator's dofs,
  projected on its coordinate (``sum(coef * load) / sum(coef^2)``; coef = 1 for a joint, the tendon's coefficients
  for a knee), minus its own tendon friction.
- **The friction budget** goes into ``dof_frictionloss`` / ``dof_damping`` for joints, and ``tendon_frictionloss`` /
  ``tendon_damping`` for tendons, per environment (``bam_servo_init`` expands all four fields). The rotor inertia goes
  on the joint or the tendon, and the tendon's XML damping is replaced.

**The supply** (``_Supply``): one per robot, shared by its actuator groups. ``vin_range`` (V) draws each env's supply
voltage once at startup (``BamActuator`` would draw one per group, as if each group had its own battery);
``vin_drop_resistance_range`` (Ohm) a pack + wiring resistance, the bus then sagging by R x the servos' total battery
current of the previous physics step (each servo's duty x motor current, summed over all twelve before clamping at zero:
regeneration is not credited; ``vin_min`` floors it); ``r_scale_range`` scales each env's winding resistance (heat),
with kt and kt_emf unchanged. All three are held across resets; none is drawn when its range is None.

**Firmware gains** (``firmware``): for ``MD300Actuator``, the configuration its ``load_log`` reads from a bench log (kp,
kd, pwm_scale, max_pwm, the frame's current cap as a duty cap, supply voltage), set once at build time.

Not in this class: the fitted ``command_delay`` (the actuator delay carries it) and the idle bridge's braking (a policy
never disables torque).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.actuator.actuator import Actuator, TransmissionType
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.spec import create_motor_actuator

from mp4_sim.actuators.bam.actuator import TorchBackend
from mp4_sim.actuators.bam.mjlab import BamActuator, BamActuatorCfg
from mp4_sim.actuators.choice import apply_md300_firmware

_FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
_FRICTION_TENDON = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)


@requires_model_fields("dof_frictionloss", "dof_damping", "tendon_frictionloss", "tendon_damping")
def bam_servo_init(env, env_ids=None) -> None:
    """Startup event: expand the friction fields BamServo writes per environment."""
    del env, env_ids


@dataclass(kw_only=True)
class BamServoCfg(BamActuatorCfg):
    r_scale_range: tuple[float, float] | None = None     # per-env winding resistance scale (heat); None: 1
    firmware: dict | None = None                         # MD300Actuator's gains (apply_md300_firmware); None: the class's

    def build(self, entity, target_ids, target_names):
        return BamServo(self, entity, target_ids, target_names)


class _Supply:
    """One robot's servo supply, per env: the voltage, the pack + wiring resistance, the winding resistance scale; and
    the bus current the groups drew on the previous physics step (for the sag)."""

    def __init__(self, cfg, n: int, device, vin_nominal: float):
        self.key = (cfg.vin_range, cfg.vin_drop_resistance_range, cfg.vin_min, cfg.r_scale_range, vin_nominal)
        if cfg.vin_range is not None:
            self.vin = torch.empty(n, 1, device=device).uniform_(*cfg.vin_range)
        else:
            self.vin = torch.full((n, 1), vin_nominal, device=device)
        self.drop = None
        if cfg.vin_drop_resistance_range is not None:
            self.drop = torch.empty(n, 1, device=device).uniform_(*cfg.vin_drop_resistance_range)
        self.r_scale = None
        if cfg.r_scale_range is not None:
            self.r_scale = torch.empty(n, 1, device=device).uniform_(*cfg.r_scale_range)
        self.vin_min = cfg.vin_min
        self.bus_current = torch.zeros(n, 1, device=device)   # A from the battery, the previous physics step
        self._pending: dict = {}

    def bus_voltage(self, who) -> torch.Tensor:
        """The supply voltage ``who`` sees this physics step. A group that has already reported a current starts a new
        step: the reports so far become the previous step's bus current."""
        if self.drop is None:
            return self.vin
        if id(who) in self._pending:
            self.bus_current = torch.stack(list(self._pending.values())).sum(0)
            self._pending = {}
        v = self.vin - self.drop * torch.clamp(self.bus_current, min=0.0)
        return v if self.vin_min is None else torch.clamp(v, min=self.vin_min)

    def report(self, who, battery_current: torch.Tensor) -> None:
        if self.drop is not None:
            self._pending[id(who)] = battery_current


def robot_supply(actuator, cfg, n: int, device, vin_nominal: float) -> _Supply:
    """The supply shared by an entity's BAM groups: made by the first group initialised on this data, reused (with the
    same ranges and nominal voltage, else an error) by the others."""
    entity, data = actuator.entity, actuator._data
    held = getattr(entity, "_bam_supply", None)
    if held is None or held[0] is not data:
        entity._bam_supply = (data, _Supply(cfg, n, device, vin_nominal))
        return entity._bam_supply[1]
    sup = held[1]
    key = (cfg.vin_range, cfg.vin_drop_resistance_range, cfg.vin_min, cfg.r_scale_range, vin_nominal)
    if key != sup.key:
        raise ValueError(f"BAM groups of one robot disagree on the supply: {key} vs {sup.key}")
    return sup


class BamServo(BamActuator):
    cfg: BamServoCfg

    def __init__(self, cfg, entity, target_ids, target_names):
        super().__init__(cfg, entity, target_ids, target_names)
        if cfg.firmware is not None:
            apply_md300_firmware(self._bam_model.actuator, cfg.firmware)

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
        pad = lambda rows, v: [r + [v] * (width - len(r)) for r in rows]
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
        self.supply = robot_supply(self, self.cfg, n, device, float(act.vin))
        self.vin_tensor = self.supply.vin
        self.vin_drop_resistance = self.supply.drop
        self._R0 = float(bam.R.value)
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
        sup = self.supply
        act.vin = sup.bus_voltage(self)
        act.kp = self._base_kp * self.kp_scale
        if sup.r_scale is not None:
            bam.R.value = self._R0 * sup.r_scale
        control = act.compute_control(cmd.position_target, pos, vel * self.kd_scale, self._dt)
        torque = act.compute_torque(control, True, pos, vel * self.kd_scale)
        if sup.r_scale is not None:
            bam.R.value = self._R0
        if sup.drop is not None:      # battery current: duty x motor current (tau / kt), summed over the group
            sup.report(self, (act.duty_cycle * torque / bam.kt.value).sum(-1, keepdim=True))

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
                      "frictionloss": budget, "target": cmd.position_target, "vin": act.vin}
        return torque
