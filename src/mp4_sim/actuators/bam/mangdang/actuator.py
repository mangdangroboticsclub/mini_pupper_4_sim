# Copyright 2026 Mangdang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import math

import numpy as np

from ..actuator import VoltageControlledActuator
from ..parameter import Parameter
from ..testbench import Testbench


class MD01PwmActuator(VoltageControlledActuator):
    """MD01 in AT32 ``control_mode 1`` (name ``md01v``): position PID -> PWM duty.

    In firmware >= 0.0.8 with ``control_mode 1`` the position loop drives the
    H-bridge duty directly. Measured at a blocked output::

        duty = clip(kp_position * kff_current * error_deg - D, +/-max_pwm)

    so ``error_gain`` = ``kff_current`` * 180/pi (duty per rad per unit kp) is
    read from every log, not fixed in the class. The plant is a brushed DC
    motor: ``x = duty * vin - kt_emf * dq`` drives the current through ``R``
    and a brush/bridge drop ``V0``, and ``tau = kt * i``. It departs from BAM's
    stock voltage law in three measured ways: a back-EMF constant ``kt_emf``
    distinct from ``kt``, a soft conduction knee instead of a hard ``V0`` dead
    band, and a braking torque while the bridge is idle. Position data alone
    do **not** pin ``kt`` and ``R`` separately -- only the combinations
    stiffness ``kt*V*G/R`` and damping ``kt*kt_emf/R + friction_viscous`` are
    visible, so the bench measurements set the ranges.

    D term: BAM has none; the firmware's ``kd_position`` is part of the
    robot's actuator (kd 400). Its form is selected by :attr:`kd_form` from
    the ramp probe in ``datasets/md01/id_3/unit_A/raw/probe`` and only scaled
    by the fit (``kd_ratio``):

    * ``"velocity"`` -- D = kd * KD_GAIN * kd_ratio * dq (stateless); this is
      what the AT32 does: in a steady ramp the error rate is ~0 while the
      velocity is not, and the duty carries a speed-proportional term, so the
      firmware differentiates the measurement, not the error;
    * ``"filtered"`` -- the same on a first-order low-pass of dq with time
      constant :attr:`KD_TAU` (state: the filtered velocity);
    * ``"error_rate"`` -- on d(error)/dt = dq_target - dq (state: the previous
      target).

    :attr:`KD_GAIN` is the probe's slope in duty per (rad/s) per unit kd; 0
    disables the term. The position frame's ``max_current_mA`` is **not** a
    current limiter in mode 1: it clips the PID output before the mA -> duty
    conversion, so it acts as a duty cap ``kff_current * cur_cap_ma``
    (measured: a 200 mA cap at 7.9 deg of error gave duty 0.200 and 113 mA,
    the plant's value at that voltage). The effective ceiling is
    therefore ``min(max_pwm, kff_current * cur_cap_ma)``; at the robot's
    1500 mA cap that is 1.5, so only ``max_pwm`` binds and nothing limits the
    current.
    """

    #: Duty per (rad/s) per unit kd_position. Measured on id_3 of unit A with
    #: constant-velocity ramps at 0.5-4 rad/s in both directions
    #: (``datasets/md01/id_3/unit_A/raw/probe/ramp_kd*.json``), pairing each
    #: kd-400 segment with its kd-0 twin and correcting for the small speed
    #: difference between them -- the measurement lag is 5x this term, so an
    #: uncorrected difference is biased: 0.0115 duty/(rad/s) at kd 400, i.e.
    #: 2.87e-5 per unit kd, stable over four settle windows (2.79-2.89e-5);
    #: the eight segment pairs spread +-15 % (2.3-3.2e-5). The fit scales it
    #: with ``kd_ratio``.
    KD_GAIN: float = 2.87e-5
    #: Velocity low-pass time constant [s] for ``kd_form = "filtered"``.
    KD_TAU: float = 0.0
    #: "velocity" | "filtered" | "error_rate" (see the class docstring).
    kd_form: str = "velocity"
    #: The identified configuration. ``load_log`` replaces it with each log's;
    #: controllers that load no log (``bam.mujoco.MujocoController``,
    #: ``bam.mjlab``) keep it.
    DEFAULT_KFF: float = 0.001
    DEFAULT_MAX_PWM: float = 0.99
    DEFAULT_KD: float = 400.0

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=12.0,
            kp=80.0,
            # Placeholder: load_log sets it from at32_kff_current.
            error_gain=0.001 * 180.0 / math.pi,
            max_pwm=0.99,
            max_current=None,
        )
        self.kd = self.DEFAULT_KD
        self.control_mode = 1
        self._state = None
        self._to_reset_env_ids: list = []

    # -- state (only used by the filtered / error_rate D forms) ---------------

    @property
    def stateful(self) -> bool:  # type: ignore[override]
        return self.kd_form != "velocity"

    def reset(self, env_ids=...) -> None:
        if env_ids is ...:
            self._state = None
            self._to_reset_env_ids.clear()
        else:
            self._to_reset_env_ids.append(env_ids)

    def get_state(self):
        return self._state, list(self._to_reset_env_ids)

    def set_state(self, state) -> None:
        s, pending = (None, []) if state is None else state
        self._state = s
        self._to_reset_env_ids = list(pending)

    # -- configuration from the log -------------------------------------------

    def load_log(self, log: dict):
        """Take the firmware configuration from the log.

        Every value may be a scalar (one log) or an array (``Logs.make_batch``
        in a fit), so nothing here casts to ``float`` and the ceiling is
        combined with ``numpy.minimum``. Missing keys fall back to the class
        defaults rather than keeping the previous log's value.
        """
        super().load_log(log)
        kff = log.get("at32_kff_current", self.DEFAULT_KFF)
        self.error_gain = kff * (180.0 / math.pi)
        max_pwm = log.get("at32_max_pwm_duty_cycle", self.DEFAULT_MAX_PWM)
        # The frame's current cap clips the position loop's output (in mA)
        # before the mA -> duty scale, so it is a duty cap, not a current
        # limit; max_pwm applies on top. At the robot's 1500 mA cap with
        # kff 0.001 this is 1.5 and only max_pwm binds.
        cap = log.get("cur_cap_ma")
        self.max_pwm = max_pwm if cap is None else np.minimum(max_pwm, cap * kff)
        self.kd = log.get("at32_kd_position", log.get("kd", self.DEFAULT_KD))
        self.control_mode = log.get("at32_control_mode", 1)
        supply = log.get("supply_v")
        if supply is not None:
            self.vin = supply
        self.max_current = None
        self.reset()

    def initialize(self):
        # Torque constant at the output [Nm/A], in reported-current units
        # (tau = kt * I as the AT32 reports I): 0.66 from the quasi-static holds
        # of datasets/md01/id_3/unit_A (plant_from_logs.py holds); the estimate
        # depends on how the static friction band is treated.
        self.model.kt = Parameter(0.68, 0.40, 1.00)

        # Back-EMF constant [V.s/rad]. **Not kt.** On the duty-saturated
        # samples of datasets/md01/id_3/unit_A, dI/d(dq) = -41.6 mA/(rad/s)
        # (plant_from_logs.py saturated), i.e. kt_emf ~ 0.42 at R ~ 10, where
        # kt_emf = kt would give ~66. A PMDC motor has kt = ke in SI units, but
        # the reported current is the motor current (a bench-supply reading at
        # duty 0.99, ~1.1 A on the display, against 1.03 A reported), so the gap
        # is not a current-scale error. The two constants are kept separate.
        self.model.kt_emf = Parameter(0.44, 0.25, 0.70)

        # Effective resistance [Ohm] at the output: 9.6-11.0 from the blocked
        # ladders at 39-43 degC, 11.3-13.2 from the in-motion regressions warm.
        self.model.R = Parameter(11.0, 8.0, 16.0)

        # Brush/bridge drop [V] and the softness of its knee. The real
        # conduction knee is soft -- 25 mA already flows at 0.74 V where a hard
        # 1.5 V dead band predicts zero -- and the residual scale of the fit
        # (5-30 mrad of error) sits inside it, so a hard band distorts exactly
        # the regime that matters. `knee_extra` keeps the small-signal
        # conductance positive by construction (see compute_current).
        self.model.V0 = Parameter(1.4, 0.5, 2.0)
        self.model.knee_extra = Parameter(1.0, 0.05, 6.0)

        # Apparent inertia at the output [kg m^2].
        self.model.armature = Parameter(3e-4, 1e-5, 5e-3)

        # Scale on the firmware gain read from the log; the law was verified to
        # ~1 % at the blocked output, so this stays tight.
        self.model.error_gain_ratio = Parameter(1.0, 0.9, 1.1)

        # Scale on the probe's D gain (the probe's segment pairs spread about
        # +-15 % around KD_GAIN).
        self.model.kd_ratio = Parameter(1.0, 0.2, 2.0)

    # -- control law and plant --------------------------------------------------

    def _velocity_estimate(self, q_target, q, dq, dt):
        """The velocity the firmware's D term multiplies, per :attr:`kd_form`."""
        if self.kd_form == "velocity":
            return dq
        state = self._state
        if state is None:
            # First call after a reset: seed from the current state, as the
            # firmware does on power-up (STS3215 pattern).
            state = dq * 0.0 if self.kd_form == "filtered" else q_target * 1.0
            self._to_reset_env_ids.clear()
        elif self._to_reset_env_ids:
            for env_ids in self._to_reset_env_ids:
                if self.kd_form == "filtered":
                    state[env_ids] = dq[env_ids] * 0.0
                else:
                    state[env_ids] = q_target[env_ids]
            self._to_reset_env_ids.clear()
        if self.kd_form == "filtered":
            alpha = dt / (self.KD_TAU + dt) if self.KD_TAU > 0 else 1.0
            state = state + alpha * (dq - state)
            self._state = state
            return state
        # error_rate: d(err)/dt = dq_target - dq  ->  D opposes -(dq_target - dq)
        dq_target = (q_target - state) / dt
        self._state = q_target
        return dq - dq_target

    def compute_control(self, q_target, q, dq, dt):
        duty = (q_target - q) * self.kp * self.error_gain * self.model.error_gain_ratio.value
        if self.KD_GAIN:
            v = self._velocity_estimate(q_target, q, dq, dt)
            duty = duty - self.kd * self.KD_GAIN * self.model.kd_ratio.value * v
        if self.max_current is not None:
            # Frame cap modelled as upstream does: a duty window around the
            # back-EMF (the firmware can only bound the duty).
            back_emf = self.model.kt.value * dq
            span = self.model.R.value * self.max_current / self.vin
            duty = self.backend.clamp(duty, (back_emf - span) / self.vin, (back_emf + span) / self.vin)
        duty = self.backend.clamp(duty, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty
        return self.vin * duty

    def compute_current(self, control, dq):
        """Motor current [A] from the applied voltage and the speed.

        ``x = duty*vin - kt_emf*dq`` is the voltage across the winding after
        back-EMF, and the conduction knee is softened as

            I = (x - V0 * x / (|x| + V0 + knee_extra)) / R

        which tends to the hard dead band ``(|x| - V0)/R`` for large ``|x|``
        and to a reduced-slope line near zero. Writing the knee scale as
        ``V0 + knee_extra`` keeps the small-signal conductance
        ``(knee_extra / (V0 + knee_extra)) / R`` positive for every parameter
        the fit can choose.
        """
        x = control - self.model.kt_emf.value * dq
        scale = self.model.V0.value + self.model.knee_extra.value
        return (x - self.model.V0.value * x / (abs(x) + scale)) / self.model.R.value

    def compute_torque(self, control, torque_enable, q, dq):
        """Motor torque [Nm], including the idle bridge's braking.

        With the bridge disabled the winding is still shorted, and the joint is
        damped at 0.032-0.042 Nm.s/rad -- ten times its mechanical drag
        (measured on the torque-off free swings of every loaded block of
        ``datasets/md01/id_3/unit_A``). Upstream's ``compute_torque`` returns
        zero when ``torque_enable`` is false, so without this term the fit has
        to hide the drop damping in ``friction_viscous``, which then over-damps
        the powered regime and drives kt, R and V0 to their bounds.

        Note: ``bam.mujoco`` and ``bam.mjlab`` always call this with
        ``torque_enable=True`` and apply no torque at all while torque is off,
        so the brake acts only in the reference simulator that ``bam.fit`` and
        ``bam.mae`` use; ``bam.mae --mujoco`` and ``--mjlab`` do not reproduce
        torque-off windows.
        """
        current = self.compute_current(control, dq)
        self.current = current
        kt = self.model.kt.value
        powered = kt * current * torque_enable
        brake = -kt * self.model.kt_emf.value * dq / self.model.R.value
        return powered + brake * (1 - torque_enable)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value


class MD01PwmPinnedActuator(MD01PwmActuator):
    """``md01v`` with every plant constant held to its measured spread (``md01vp``).

    Position data see only combinations (stiffness ``kt*V*G/R``, damping
    ``kt*kt_emf/R + friction_viscous``), so a fit with a loose plant walks along
    those directions until it hits a bound: fitted loose on the id_3 dataset it
    reached kt 0.21 / R 5.0 / V0 0 and drew 3.4x the measured current. These ranges are the bench
    measurements' own spread: kt from the holds and the dynamic torque balance,
    ``kt_emf`` from the duty-saturated samples, R and V0 from the blocked
    ladders (cold) and the in-motion regressions (warm).
    """

    def initialize(self):
        super().initialize()
        self.model.kt = Parameter(0.68, 0.60, 0.80)
        self.model.kt_emf = Parameter(0.44, 0.38, 0.52)
        self.model.R = Parameter(11.5, 9.5, 13.5)
        self.model.V0 = Parameter(1.45, 1.20, 1.70)
        self.model.knee_extra = Parameter(1.0, 0.05, 4.0)
        self.model.error_gain_ratio = Parameter(1.0, 0.95, 1.05)
        self.model.kd_ratio = Parameter(1.0, 0.5, 1.5)
