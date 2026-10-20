"""The heading loop's law gives the same numbers on torch tensors (the evaluation tools) as on floats (the robot)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mp4_sim.heading import pi_step  # noqa: E402


def test_torch_matches_numpy():
    rng = np.random.default_rng(0)
    errs = rng.uniform(-0.8, 0.8, (300, 64))
    for aw in ("none", "cond"):
        for wz_max in (0.4, 0.6):
            integ_t = torch.zeros(64, dtype=torch.float64)
            integ_f = [0.0] * 64
            for row in errs:
                wz_t, integ_t = pi_step(torch.as_tensor(row), integ_t, 0.02, wz_max=wz_max, antiwindup=aw)
                for i, e in enumerate(row):
                    wz_f, integ_f[i] = pi_step(float(e), integ_f[i], 0.02, wz_max=wz_max, antiwindup=aw)
                    integ_f[i] = float(integ_f[i])
                    assert abs(float(wz_f) - float(wz_t[i])) < 1e-12
            assert np.allclose(integ_t.numpy(), integ_f, atol=1e-12)


def test_float32_parity():
    """The tools run float32 on the GPU: the same law agrees with float64 to float32 precision."""
    rng = np.random.default_rng(1)
    errs = rng.uniform(-0.8, 0.8, (500, 32))
    i32, i64 = torch.zeros(32, dtype=torch.float32), torch.zeros(32, dtype=torch.float64)
    for row in errs:
        w32, i32 = pi_step(torch.as_tensor(row, dtype=torch.float32), i32, 0.02, antiwindup="cond")
        w64, i64 = pi_step(torch.as_tensor(row), i64, 0.02, antiwindup="cond")
        assert torch.allclose(w32.double(), w64, atol=1e-5)
