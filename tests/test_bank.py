"""The CPU servo bank (``mp4_sim.actuators.bank``) on SKU2."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mp4_sim.actuators import bank, fits
from mp4_sim.contract import Contract
from mp4_sim.robot import sku2


def _robot(extra_lag_steps=0):
    models = bank.load_models(fits.T5UB06)
    spec = sku2.get_spec()
    bank.add_servos(spec, models)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.1])
    m = spec.compile()
    m.opt.timestep = fits.PHYSICS_DT
    d = mujoco.MjData(m)
    c = Contract.load()
    for name, q in zip(c.joint_order, c.stand_joint_rad):
        d.qpos[m.joint(name).qposadr[0]] = q
    d.qpos[2] = sku2.INIT_BASE_Z
    mujoco.mj_forward(m, d)
    b = bank.BamServoBank(m, d, models, extra_lag_steps=extra_lag_steps)
    ids = [m.actuator(n).id for n in fits.SERVO_TARGETS]
    return m, d, b, d.actuator_length[ids].copy(), ids


def test_servos_added():
    m, _, b, _, _ = _robot()
    assert m.nu == 12
    assert [m.actuator(i).name for i in range(m.nu)] == list(fits.SERVO_TARGETS)
    for n in fits.SERVO_TARGETS:
        a = m.actuator(n)
        assert not a.forcelimited and not a.ctrllimited
    assert {r: len(x.names) for r, x in b.roles.items()} == {"abd": 4, "hip": 4, "knee": 4}


def test_holds_the_stand():
    m, d, b, stand, ids = _robot()
    b.set_targets(stand)
    for _ in range(1000):
        b.update()
        mujoco.mj_step(m, d)
    assert 0.06 < d.qpos[2] < 0.08
    assert d.xmat[1][8] > 0.99                                   # upright
    assert np.abs(d.actuator_length[ids] - stand).max() < np.radians(3.0)
    assert np.all(np.isfinite(b.last["torque"]))
    assert np.all(b.last["frictionloss"] > 0.0)


@pytest.mark.parametrize("extra", [0, 3])
def test_command_latency(extra):
    m, d, b, stand, _ = _robot(extra_lag_steps=extra)
    assert b.latency_steps == {"abd": 1 + extra, "hip": 5 + extra, "knee": 1 + extra}
    b.set_targets(stand)
    for _ in range(10):
        b.update()
        mujoco.mj_step(m, d)
    step = stand + 0.1
    b.set_targets(step)
    first = {}
    for k in range(12):
        b.update()
        mujoco.mj_step(m, d)
        for role, r in b.roles.items():
            if role not in first and np.allclose(b.last["target"][r.rows], step[r.rows]):
                first[role] = k
    assert first == b.latency_steps


def test_reset_fills_history_with_first_command():
    m, d, b, stand, _ = _robot()
    b.set_targets(stand + 0.05)
    b.update()
    assert np.allclose(b.last["target"], stand + 0.05)


def test_friction_budget_matches_gpu():
    pytest.importorskip("mjlab")
    torch = pytest.importorskip("torch")
    from mp4_sim.actuators.bam.mjlab import BamActuator

    rng = np.random.default_rng(0)
    motor, external = rng.normal(0, 0.2, (2, 64))
    vel = rng.normal(0, 2.0, 64)
    for role, path in fits.T5UB06.items():
        model = fits.load(path)
        stribeck = np.exp(-np.power(np.abs(vel) / model.dtheta_stribeck.value, model.alpha.value))
        fake = SimpleNamespace(_bam_model=model, friction_scale=None)
        gpu = BamActuator._compute_friction_budget(fake, torch.tensor(motor), torch.tensor(external),
                                                   torch.tensor(stribeck)).numpy()
        np.testing.assert_allclose(bank.friction_budget(model, motor, external, vel), gpu, rtol=1e-12,
                                   err_msg=role)
