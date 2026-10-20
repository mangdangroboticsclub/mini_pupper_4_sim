"""The BAM servos on plain MuJoCo (CPU): the same arithmetic as the GPU's ``BamServo``, one robot at a time.

Each physics step, before ``mj_step``, :meth:`BamServoBank.update` does what ``BamServo.compute`` does, per role:

- the servo's position and velocity are ``actuator_length`` / ``actuator_velocity`` (the previous step's forward
  pass), for joints and the knee's tendon alike;
- the firmware law (``compute_control``) and the motor plant (``compute_torque``) give the torque written to
  ``ctrl``, with ``dt`` the model's timestep;
- the external load is ``-qfrc_bias + qfrc_constraint - (MuJoCo's own dof friction)`` on the servo's dofs, projected
  on its coordinate (``sum(coef * load) / sum(coef^2)``), minus its own tendon friction;
- the friction budget (:func:`friction_budget`, the GPU's formula) against the servo's own ``actuator_force`` goes
  into ``dof_frictionloss`` / ``dof_damping`` or ``tendon_frictionloss`` / ``tendon_damping``.

The command latency is applied here too: each role's fitted ``command_delay`` (``fits.command_delay_steps``) plus
``extra_lag_steps`` (the environment's sensing-to-command chain), held in physics steps as mjlab's delay buffer does
(right after a reset the oldest command stands in for the missing history).

Usage::

    models = load_models(fits.T5UB06)
    spec = sku2.get_spec()
    add_servos(spec, models)
    m = spec.compile(); m.opt.timestep = fits.PHYSICS_DT
    d = mujoco.MjData(m)
    bank = BamServoBank(m, d, models)
    bank.set_targets(targets)          # per servo, in fits.SERVO_TARGETS order, the actuator's own coordinate
    bank.update(); mujoco.mj_step(m, d)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import fits

_FRICTION_DOF = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF)
_FRICTION_TENDON = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)


def load_models(params: dict, scales: tuple = ()) -> dict:
    """role -> a fresh BAM model, from role -> fitted JSON (e.g. ``fits.T5UB06``)."""
    models = {role: fits.load(params[role], scales) for role in fits.ROLES}
    for role, model in models.items():
        model.params_path = params[role]
    return models


def add_servos(spec: mujoco.MjSpec, models: dict, targets: tuple = fits.SERVO_TARGETS) -> None:
    """Add one motor actuator per servo target, as ``BamActuator`` / ``BamServo`` edit the spec for mjlab.

    No force or control limits (the BAM model bounds the torque); the role's rotor inertia as the joint's or the
    tendon's armature; damping and frictionloss zeroed (``update`` writes them every step).
    """
    for name in targets:
        armature = float(models[fits.role_of(name)].actuator.get_extra_inertia())
        tendon = fits.role_of(name) == "knee"
        act = spec.add_actuator(name=name, target=name)
        act.trntype = mujoco.mjtTrn.mjTRN_TENDON if tendon else mujoco.mjtTrn.mjTRN_JOINT
        act.dyntype = mujoco.mjtDyn.mjDYN_NONE
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.biastype = mujoco.mjtBias.mjBIAS_NONE
        act.gear = [1.0, 0, 0, 0, 0, 0]
        act.forcelimited = False
        act.ctrllimited = False
        target = spec.tendon(name) if tendon else spec.joint(name)
        target.armature = armature
        target.damping = np.zeros((3, 1))
        target.frictionloss = 0.0


def friction_budget(model, motor: np.ndarray, external: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """The velocity-independent friction budget, as ``bam.mjlab``'s ``_compute_friction_budget`` computes it.

    It differs from ``Model.compute_frictions`` in the m6 quadratic term: that one applies it only when the motor
    and the external torque have opposite signs, the GPU path (and so this) always does.
    """
    m = model
    budget = np.full_like(motor, m.friction_base.value, dtype=float)
    stribeck = (np.exp(-np.power(np.abs(vel) / m.dtheta_stribeck.value, m.alpha.value)) if m.stribeck
                else np.zeros_like(vel))
    if m.stribeck:
        budget = budget + stribeck * m.friction_stribeck.value
    if m.load_dependent:
        if m.directional:
            budget = budget + np.abs(external * m.load_friction_external.value
                                     - motor * m.load_friction_motor.value)
            if m.stribeck:
                budget = budget + stribeck * np.abs(external * m.load_friction_external_stribeck.value
                                                    - motor * m.load_friction_motor_stribeck.value)
                if m.quadratic:
                    drive = (np.abs(motor) > np.abs(external)).astype(float)
                    budget = budget + stribeck * (drive * m.load_friction_external_quad.value * external ** 2
                                                  + (1.0 - drive) * m.load_friction_motor_quad.value * motor ** 2)
        else:
            gearbox = np.abs(external - motor)
            budget = budget + m.load_friction_base.value * gearbox
            if m.stribeck:
                budget = budget + stribeck * m.load_friction_stribeck.value * gearbox
    return budget


@dataclass
class _Role:
    model: object
    names: list
    rows: np.ndarray                  # positions in the bank's target order
    act: np.ndarray                   # actuator ids
    dofs: np.ndarray                  # (J, W), padded with 0
    coefs: np.ndarray                 # (J, W), padded with 0
    norm: np.ndarray                  # (J,)
    tendons: np.ndarray               # (J,), -1 for a joint
    lag: int
    history: deque = field(default_factory=deque)

    @property
    def is_tendon(self) -> bool:
        return bool(self.tendons[0] >= 0)


class BamServoBank:
    """The servos of one robot. ``models`` is :func:`load_models`'s, the same instances :func:`add_servos` used.

    Domain randomisation, as on the GPU: ``kp_scale`` and ``kd_scale`` (the kd scale multiplies the velocity the
    firmware law and the plant see), ``vin`` (supply voltage, V) and ``friction_scale`` (the whole budget).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, models: dict,
                 targets: tuple = fits.SERVO_TARGETS, extra_lag_steps: int = 0):
        self.model, self.data = model, data
        self.targets = tuple(targets)
        self.dt = float(model.opt.timestep)
        self.kp_scale = 1.0
        self.kd_scale = 1.0
        self.friction_scale = 1.0
        self.roles: dict[str, _Role] = {}
        for role in fits.ROLES:
            rows = [k for k, n in enumerate(self.targets) if fits.role_of(n) == role]
            if rows:
                self.roles[role] = self._build(role, models[role], rows)
        bam = next(iter(self.roles.values())).model.actuator
        self.vin = float(bam.vin)
        self._base_kp = {role: float(r.model.actuator.kp) for role, r in self.roles.items()}
        self.command = np.zeros(len(self.targets))
        self.last: dict[str, np.ndarray] = {}
        self.set_extra_lag(extra_lag_steps)
        self.reset()

    def _build(self, role: str, bam, rows: list) -> _Role:
        m = self.model
        names = [self.targets[k] for k in rows]
        act, dofs, coefs, tendons = [], [], [], []
        for name in names:
            a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if a < 0:
                raise ValueError(f"no actuator {name!r}: build the model with add_servos()")
            if int(m.actuator_biastype[a]) != int(mujoco.mjtBias.mjBIAS_NONE) or m.actuator_gainprm[a, 0] != 1.0:
                raise ValueError(f"actuator {name!r} is not a motor (add_servos() makes them)")
            trn = int(m.actuator_trntype[a])
            tid = int(m.actuator_trnid[a, 0])
            if trn == int(mujoco.mjtTrn.mjTRN_TENDON):
                adr, num = m.tendon_adr[tid], m.tendon_num[tid]
                dofs.append([int(m.jnt_dofadr[m.wrap_objid[k]]) for k in range(adr, adr + num)])
                coefs.append([float(m.wrap_prm[k]) for k in range(adr, adr + num)])
                tendons.append(tid)
            else:
                dofs.append([int(m.jnt_dofadr[tid])])
                coefs.append([1.0])
                tendons.append(-1)
            act.append(a)
        width = max(len(d) for d in dofs)
        dofs = np.array([d + [0] * (width - len(d)) for d in dofs], dtype=int)
        coefs = np.array([c + [0.0] * (width - len(c)) for c in coefs])
        lag = fits.command_delay_steps(bam.params_path)
        return _Role(bam, names, np.array(rows), np.array(act), dofs, coefs, (coefs ** 2).sum(-1),
                     np.array(tendons), lag)

    # ------------------------------------------------------------------------------------------------ interface
    @property
    def latency_steps(self) -> dict:
        """role -> the command latency in physics steps (fitted + extra)."""
        return {role: r.lag + self.extra_lag_steps for role, r in self.roles.items()}

    def set_extra_lag(self, steps: int) -> None:
        """The environment's chain delay, added to every role's fitted delay (set it at reset)."""
        self.extra_lag_steps = int(steps)
        for role, r in self.roles.items():
            r.history = deque(r.history, maxlen=r.lag + self.extra_lag_steps + 1)

    def set_targets(self, targets) -> None:
        """The commanded position per servo, in :attr:`targets` order: joint angle or tendon length (rad)."""
        self.command = np.array(targets, dtype=float).reshape(len(self.targets))

    def reset(self) -> None:
        """Forget the command history and the firmware's state (after the joints are teleported)."""
        for r in self.roles.values():
            r.history.clear()
            r.model.reset()

    def update(self) -> None:
        """One evaluation: torque into ``ctrl``, the friction budget into the model. Call before ``mj_step``."""
        m, d = self.model, self.data
        bias, constraint = d.qfrc_bias, d.qfrc_constraint
        f_dof, f_tendon = self._efc_friction()
        load = -bias + constraint - f_dof
        for role, r in self.roles.items():
            r.history.append(self.command[r.rows].copy())
            target = r.history[0]          # maxlen is lag + 1: the command from lag steps ago, or the oldest
            act = r.model.actuator
            pos = d.actuator_length[r.act].copy()
            vel = d.actuator_velocity[r.act].copy()
            act.vin = self.vin
            act.kp = self._base_kp[role] * self.kp_scale
            control = act.compute_control(target, pos, vel * self.kd_scale, self.dt)
            torque = act.compute_torque(control, True, pos, vel * self.kd_scale)
            d.ctrl[r.act] = torque

            external = (load[r.dofs] * r.coefs).sum(-1) / r.norm
            if r.is_tendon:
                external = external - f_tendon[r.tendons]
            motor = d.actuator_force[r.act].copy()
            budget = friction_budget(r.model, motor, external, vel) * self.friction_scale
            viscous = r.model.friction_viscous.value
            if r.is_tendon:
                m.tendon_frictionloss[r.tendons] = budget
                m.tendon_damping[r.tendons] = viscous
            else:
                m.dof_frictionloss[r.dofs[:, 0]] = budget
                m.dof_damping[r.dofs[:, 0]] = viscous
            for key, value in (("target", target), ("pos", pos), ("vel", vel), ("torque", torque),
                               ("motor", motor), ("external", external), ("frictionloss", budget)):
                self.last.setdefault(key, np.zeros(len(self.targets)))[r.rows] = value

    def _efc_friction(self) -> tuple[np.ndarray, np.ndarray]:
        """The forces MuJoCo's own friction constraints apply, per dof and per tendon."""
        m, d = self.model, self.data
        f_dof, f_tendon = np.zeros(m.nv), np.zeros(max(m.ntendon, 1))
        n = int(d.nefc)
        if n:
            typ, ids, force = d.efc_type[:n], d.efc_id[:n], d.efc_force[:n]
            sel = typ == _FRICTION_DOF
            np.add.at(f_dof, ids[sel], force[sel])
            sel = typ == _FRICTION_TENDON
            np.add.at(f_tendon, ids[sel], force[sel])
        return f_dof, f_tendon
