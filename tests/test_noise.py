"""The observation-noise table: per-element amplitudes (noise_n2.json), the ramp's end values, the named profiles."""
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
import torch  # noqa: E402

from mp4_sim.gpu import mdp  # noqa: E402
from mp4_sim.gpu.env_cfg import NOISE_END, NOISE_START, noise_profile  # noqa: E402


def _env(step=0):
    return SimpleNamespace(device="cpu", common_step_counter=step)


def test_profiles():
    assert noise_profile("n1") == NOISE_END
    n2, n2x2 = noise_profile("n2"), noise_profile("n2x2")
    assert [len(n2[k]) for k in mdp.NOISY] == [3, 3, 12, 12]
    for k in mdp.NOISY:
        assert n2x2[k] == pytest.approx([2.0 * v for v in n2[k]])
        assert min(n2[k]) >= NOISE_START[k] - 1e-9          # never below the clean floor the ramp starts from
    with pytest.raises(ValueError):
        noise_profile("n3")


def test_per_element_amplitude_bounds_the_noise():
    end = noise_profile("n2")
    env = _env()
    env.sim_noise = mdp.NoiseTable(env, NOISE_START, end)
    env.sim_noise.amp = dict(env.sim_noise.end)
    torch.manual_seed(0)
    x = torch.zeros(20000, 12)
    y = env.sim_noise.apply("vel", x)
    amp = torch.tensor(end["vel"])
    assert (y.abs() <= amp + 1e-6).all()
    assert torch.allclose(y.abs().amax(0), amp, rtol=0.01)  # each element fills its own band
    assert torch.allclose(y.std(0), amp / 3 ** 0.5, rtol=0.03)


def test_ramp_reaches_the_per_element_end():
    end = noise_profile("n2")
    env = _env()
    env.sim_noise = mdp.NoiseTable(env, NOISE_START, end)
    env.common_step_counter = 24 * 150
    assert mdp.ramp_noise(env, None, 150, 300) == 0.0
    assert torch.allclose(env.sim_noise.amp["pos"], torch.full((12,), NOISE_START["pos"]))
    env.common_step_counter = 24 * 450
    assert mdp.ramp_noise(env, None, 150, 300) == 1.0
    for k in mdp.NOISY:
        assert torch.allclose(env.sim_noise.amp[k], torch.tensor(end[k]))


def test_scalar_profile_still_works():
    env = _env()
    t = mdp.NoiseTable(env, NOISE_START, NOISE_END)
    t.amp = dict(t.end)
    y = t.apply("ang_vel", torch.zeros(1000, 3))
    assert y.abs().max() <= NOISE_END["ang_vel"] + 1e-6
