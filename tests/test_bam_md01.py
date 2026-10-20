"""Offline checks of the mode-1 MD01 class (``md01v``): law, plant, D forms, closure."""

import json
import math

import numpy as np

from mp4_sim.actuators.bam import simulate
from mp4_sim.actuators.bam.actuators import actuators
from mp4_sim.actuators.bam.mangdang.actuator import MD01PwmActuator
from mp4_sim.actuators.bam.model import load_model_from_dict, models
from mp4_sim.actuators.bam.testbench import Pendulum

KFF = 0.001
LOG_META = {
    "mass": 0.121, "arm_mass": 0.016, "length": 0.10, "kp": 80.0, "kd": 0.0, "vin": 12.0,
    "motor": "md01", "trajectory": "sin_sin", "at32_kff_current": KFF,
    "at32_max_pwm_duty_cycle": 0.99, "at32_kd_position": 0.0, "at32_control_mode": 1,
    "supply_v": 12.0, "cur_cap_ma": 1500, "dt": 0.005, "entries": [],
}


def make_model(kt=0.68, R=11.0, V0=1.4, kd_form="velocity", kd_gain=0.0, **friction):
    model = models["m1"]()
    act = MD01PwmActuator(Pendulum)
    act.kd_form = kd_form
    act.KD_GAIN = kd_gain
    model.set_actuator(act)
    model.kt.value, model.R.value, model.V0.value = kt, R, V0
    model.kt_emf.value = 0.44
    model.knee_extra.value = 1.0
    model.armature.value = 3e-4
    model.error_gain_ratio.value = 1.0
    model.kd_ratio.value = 1.0
    model.friction_base.value = friction.get("friction_base", 0.008)
    model.friction_viscous.value = friction.get("friction_viscous", 0.004)
    model.q_offset.value = 0.0
    model.command_delay.value = 0.0
    return model


def test_gain_and_limits_come_from_the_log():
    model = make_model()
    act = model.actuator
    act.load_log({**LOG_META, "kd": 400.0, "at32_kd_position": 400.0, "at32_max_pwm_duty_cycle": 0.6})
    assert abs(act.error_gain - KFF * 180 / math.pi) < 1e-12
    assert act.max_pwm == 0.6 and act.kd == 400.0 and act.kp == 80.0 and act.vin == 12.0
    # duty = kp * kff * error_deg: 80 * 0.001 * 5 deg = 0.4 -> 4.8 V
    volts = act.compute_control(math.radians(5.0), 0.0, 0.0, 0.005)
    assert abs(volts - 12.0 * 0.4) < 1e-9
    # saturation at max_pwm
    assert abs(act.compute_control(math.radians(30.0), 0.0, 0.0, 0.005) - 12.0 * 0.6) < 1e-9


def test_plant_knee_is_soft_and_back_emf_uses_its_own_constant():
    model = make_model(kt=0.68, R=11.0, V0=1.4)
    act = model.actuator
    model.kt_emf.value, model.knee_extra.value = 0.44, 1.0
    act.load_log(LOG_META)
    # Soft knee: current flows below V0, but less than a drop-free plant gives.
    i_small = act.compute_current(0.74, 0.0)
    assert 0.0 < i_small < 0.74 / 11.0
    # Far above the knee it tends to the hard form (|x| - V0)/R; the knee
    # decays as V0*scale/x^2, so it is still worth ~24 mA at 10 V.
    assert abs(act.compute_current(10.2, 0.0) - (10.2 - 1.4) / 11.0) < 0.03
    assert abs(act.compute_current(60.0, 0.0) - (60.0 - 1.4) / 11.0) < 0.01
    # Small-signal conductance is positive for every allowed parameter choice.
    for v0, extra in ((2.0, 0.05), (0.5, 6.0), (1.4, 1.0)):
        model.V0.value, model.knee_extra.value = v0, extra
        assert act.compute_current(1e-4, 0.0) > 0.0
    model.V0.value, model.knee_extra.value = 1.4, 1.0
    # Back-EMF uses kt_emf, not kt: doubling kt_emf doubles the speed term.
    base = act.compute_current(6.0, 0.0)
    model.kt_emf.value = 0.44
    one = base - act.compute_current(6.0, 1.0)
    model.kt_emf.value = 0.88
    two = base - act.compute_current(6.0, 1.0)
    assert abs(two - 2 * one) < 0.02 * abs(two)


def test_idle_bridge_brakes():
    """With torque off the winding is still shorted: tau = -kt*kt_emf*dq/R."""
    model = make_model(kt=0.68, R=11.0, V0=1.4)
    act = model.actuator
    model.kt_emf.value = 0.44
    act.load_log(LOG_META)
    tau = act.compute_torque(0.0, False, 0.0, 2.0)
    assert abs(tau - (-0.68 * 0.44 * 2.0 / 11.0)) < 1e-12
    assert tau < 0                                   # opposes the motion
    assert act.compute_torque(0.0, False, 0.0, 0.0) == 0.0
    # Powered, the brake term is absent and the motor torque is kt * I.
    powered = act.compute_torque(12.0 * 0.5, True, 0.0, 0.0)
    assert abs(powered - 0.68 * act.compute_current(6.0, 0.0)) < 1e-12


def test_d_term_forms():
    kd_gain = 2e-4  # duty per (rad/s) per unit kd
    dt = 0.005
    # velocity form: duty = P - kd*gain*dq
    model = make_model(kd_form="velocity", kd_gain=kd_gain)
    act = model.actuator
    act.load_log({**LOG_META, "at32_kd_position": 400.0})
    p_only = act.compute_control(0.1, 0.0, 0.0, dt)
    with_d = act.compute_control(0.1, 0.0, 1.0, dt)
    assert abs((p_only - with_d) / 12.0 - 400 * kd_gain * 1.0) < 1e-12
    assert act.stateful is False
    # filtered form converges to the velocity form
    model = make_model(kd_form="filtered", kd_gain=kd_gain)
    act = model.actuator
    act.KD_TAU = 0.01
    act.load_log({**LOG_META, "at32_kd_position": 400.0})
    assert act.stateful is True
    for _ in range(200):
        v = act.compute_control(0.1, 0.0, 1.0, dt)
    assert abs(v - with_d) < 1e-9
    # error_rate form: with a constant target it equals the velocity form
    model = make_model(kd_form="error_rate", kd_gain=kd_gain)
    act = model.actuator
    act.load_log({**LOG_META, "at32_kd_position": 400.0})
    act.compute_control(0.1, 0.0, 1.0, dt)
    assert abs(act.compute_control(0.1, 0.0, 1.0, dt) - with_d) < 1e-9
    # and a moving target enters through its rate: target rate 1 rad/s cancels dq 1 rad/s
    act.reset()
    act.compute_control(0.0, 0.0, 1.0, dt)
    p_small = 12.0 * 80.0 * act.error_gain * (1.0 * dt)
    assert abs(act.compute_control(1.0 * dt, 0.0, 1.0, dt) - p_small) < 1e-9


def synthetic_log(model, seed=0):
    rng = np.random.default_rng(seed)
    dt = 0.005
    t = np.arange(0.0, 6.0, dt)
    goal = 0.8 * np.sin(2 * np.pi * 0.7 * t) * np.sin(2 * np.pi * 0.11 * t + 1.0)
    enable = np.ones_like(t, dtype=bool)
    enable[(t > 2.0) & (t < 3.0)] = False  # a free-fall window
    entries = [{"timestamp": float(ti), "goal_position": float(g), "torque_enable": bool(e),
                "position": 0.0, "speed": 0.0, "load": 0.0} for ti, g, e in zip(t, goal, enable)]
    log = {**LOG_META, "entries": entries}
    sim = simulate.Simulator(model)
    pos, vel, ctrl = sim.rollout_log(log, simulate_control=True)
    for e, p, v in zip(log["entries"], pos, vel):
        e["position"] = float(p) + float(rng.normal(0, 0.0005))
        e["speed"] = float(v)
    return log


def mae(model, log):
    sim = simulate.Simulator(model)
    pos, _, _ = sim.rollout_log(log, simulate_control=True)
    return float(np.mean(np.abs(np.array(pos) - np.array([e["position"] for e in log["entries"]]))))


def test_kt_and_R_are_degenerate_in_position_but_not_in_current():
    """Scaling kt and R together leaves every torque unchanged.

    duty does not depend on the plant; I scales as 1/R; tau = kt*I and the
    brake -kt*kt_emf*dq/R both scale as kt/R. So position data see only kt/R
    (and, through kt*kt_emf/R, the damping) -- the current is what sets the
    scale, which is why the bench measurements set the ranges (md01vp).
    """
    truth = make_model(kt=0.68, R=11.0, V0=1.4)
    log = synthetic_log(truth)
    base = mae(truth, log)
    scaled = make_model(kt=0.68 * 1.3, R=11.0 * 1.3, V0=1.4)
    assert mae(scaled, log) < base + 1e-3          # indistinguishable in position
    # ... while the current it predicts differs by the same factor.
    for m in (truth, scaled):
        m.actuator.load_log(LOG_META)
    i_truth = truth.actuator.compute_current(6.0, 0.0)
    i_scaled = scaled.actuator.compute_current(6.0, 0.0)
    assert abs(i_truth / i_scaled - 1.3) < 0.1


def test_back_emf_constant_and_knee_are_visible_in_position():
    truth = make_model(kt=0.68, R=11.0, V0=1.4)
    log = synthetic_log(truth)
    base = mae(truth, log)
    softer = make_model(kt=0.68, R=11.0, V0=1.4)
    softer.kt_emf.value = 0.25                      # less damping
    assert mae(softer, log) > 3 * base
    harder = make_model(kt=0.68, R=11.0, V0=1.4)
    harder.knee_extra.value = 0.05                  # nearly a hard dead band
    assert mae(harder, log) > 2 * base


def test_round_trip_through_json_and_registry():
    truth = make_model(kt=0.7, R=12.0, V0=1.1)
    data = {"model": "m1", "actuator": "md01v", **truth.get_parameter_values()}
    model = load_model_from_dict(json.loads(json.dumps(data)))
    assert isinstance(model.actuator, MD01PwmActuator)
    assert abs(model.V0.value - 1.1) < 1e-12
    assert actuators["md01v"]().control_unit() == "volts"


def test_load_log_survives_a_batch_and_does_not_keep_stale_values():
    """bam.fit loads all logs as one batch, so every per-log value is an array."""
    model = make_model()
    act = model.actuator
    n = 3
    batch = {**LOG_META,
             "mass": np.full(n, 0.121), "arm_mass": np.full(n, 0.016), "length": np.full(n, 0.10),
             "kp": np.full(n, 80.0), "vin": np.full(n, 12.0), "dt": np.full(n, 0.005),
             "at32_kff_current": np.full(n, KFF), "at32_max_pwm_duty_cycle": np.full(n, 0.99),
             "at32_kd_position": np.full(n, 400.0), "at32_control_mode": np.full(n, 1.0),
             "cur_cap_ma": np.full(n, 1500.0), "supply_v": np.full(n, 12.0)}
    act.load_log(batch)
    assert np.allclose(act.error_gain, KFF * 180 / math.pi)
    assert np.allclose(act.max_pwm, 0.99)        # min(0.99, 1500 * 0.001 = 1.5)
    duty = act.compute_control(np.full(n, math.radians(5.0)), np.zeros(n), np.zeros(n), 0.005) / 12.0
    assert np.allclose(duty, 0.4, atol=1e-9)

    # a low frame cap becomes a duty ceiling ...
    act.load_log({**LOG_META, "cur_cap_ma": 200})
    assert abs(act.max_pwm - 0.2) < 1e-12
    # ... and does not survive into the next log
    act.load_log(LOG_META)
    assert abs(act.max_pwm - 0.99) < 1e-12


def test_kd_gain_matches_the_bench_measurement():
    """0.0115 duty per rad/s at kd 400 (ramp probe, 2026-09-22)."""
    model = make_model(kd_form="velocity", kd_gain=MD01PwmActuator.KD_GAIN)
    act = model.actuator
    act.load_log({**LOG_META, "at32_kd_position": 400.0})
    p = act.compute_control(0.1, 0.0, 0.0, 0.005)
    d = act.compute_control(0.1, 0.0, 1.0, 0.005)
    assert abs((p - d) / 12.0 - 0.0115) < 2e-4
