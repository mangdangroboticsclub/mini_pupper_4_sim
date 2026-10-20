"""The BAM actuator model's parameter echo: what the compiled model and the actuators really run (GPU)."""
import pytest

pytest.importorskip("mjlab")
import mujoco  # noqa: E402
import torch  # noqa: E402

from mp4_sim.actuators import arms, fits  # noqa: E402
from mp4_sim.gpu.env_cfg import TaskOptions, make_env_cfg  # noqa: E402

N = 8
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")


@pytest.fixture(scope="module")
def env():
    from mjlab.envs import ManagerBasedRlEnv
    e = ManagerBasedRlEnv(cfg=make_env_cfg(TaskOptions(arm="bam_t5ub06", num_envs=N)), device="cuda:0")
    e.reset()
    yield e
    e.close()


def test_bam_actuators_echo(env):
    m = env.sim.mj_model
    arm = arms.get("bam_t5ub06")
    acts = env.scene["robot"].actuators
    assert [type(a).__name__ for a in acts] == ["BamServo"] * 3
    J = {role: float(fits.load(fits.T5UB06[role]).actuator.get_extra_inertia()) for role in fits.ROLES}
    for leg in ("FR", "FL", "RR", "RL"):
        for joint, role in (("hip", "abd"), ("thigh", "hip")):
            dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{leg}_{joint}")]
            assert m.dof_armature[dof] == pytest.approx(J[role])
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, f"robot/{leg}_knee_drive")
        assert m.tendon_armature[t] == pytest.approx(J["knee"])
        assert m.tendon_damping[t] == 0.0 and m.tendon_frictionloss[t] == 0.0
    for i in range(m.nu):                                   # torque actuators without limits
        assert m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_NONE
        assert not m.actuator_forcelimited[i] and not m.actuator_ctrllimited[i]
    for a in acts:                                          # the identified configuration
        act = a._bam_model.actuator                         # kp and vin become per-env tensors while stepping
        assert (a._base_kp, act.kd, act.max_pwm) == (80.0, 400.0, 0.99)
        assert torch.all(a.vin_tensor == 12.0)
        assert not a.cfg.stiff_frictionloss
    lat = [int(a._delay_buffer.current_lags[0]) for a in acts]
    chain = lat[0] - arm.latency_steps[0]
    assert [x - chain for x in lat] == list(arm.latency_steps) == [1, 5, 1]
    assert 8 <= chain <= 19


def test_bam_writes_tendon_friction_per_env(env):
    a = torch.zeros(N, 12, device="cuda:0")
    for _ in range(10):
        env.step(a)
    model = env.sim.model
    fl = model.tendon_frictionloss
    assert fl.stride(0) != 0                                # expanded per world
    knee = env.scene["robot"].actuators[2]
    vals = fl[:, knee._tendons]
    assert torch.all(vals > 0)                              # the BAM budget, not the XML's 0
