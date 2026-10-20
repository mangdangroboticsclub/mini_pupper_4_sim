"""ONNX export: the graph reproduces the actor, the manifest is complete and clean, and every shipped policy verifies."""
import json

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from mp4_sim import paths  # noqa: E402
from mp4_sim.export import manifest as manifest_mod  # noqa: E402
from mp4_sim.export import onnx as onnx_mod  # noqa: E402
from mp4_sim.export.verify import verify  # noqa: E402


def _layers(n_in, seed=0, sizes=(128, 128, 64, 12)):
    rng = np.random.default_rng(seed)
    w, b, prev = [], [], n_in
    for n in sizes:
        w.append(rng.normal(0.0, 1.0 / np.sqrt(prev), (n, prev)).astype(np.float32))
        b.append(rng.normal(0.0, 0.1, n).astype(np.float32))
        prev = n
    return w, b


@pytest.mark.parametrize("history", [1, 10])
def test_round_trip_matches_the_actor(tmp_path, history):
    """policy.onnx on onnxruntime gives the torch actor's outputs; load_policy returns the same weights and the
    manifest; verify passes."""
    torch = pytest.importorskip("torch")
    w, b = _layers(45 * history)
    m = manifest_mod.build([0.25, 0.5, 0.5] * 4, omni=True, history=history,
                           command=manifest_mod.heading_loop_command(), task={"name": "terrain", "actuator": "bam_t5ub06"})
    sd = {}
    for i, (wi, bi) in enumerate(zip(w, b)):
        sd[f"mlp.{2 * i}.weight"], sd[f"mlp.{2 * i}.bias"] = torch.from_numpy(wi), torch.from_numpy(bi)
    ck = tmp_path / "model_0.pt"
    torch.save({"actor_state_dict": sd}, ck)
    out = tmp_path / "pol"
    onnx_mod.export(ck, out, [0.25, 0.5, 0.5] * 4, history=history, omni=True, command=m["command"], task=m["task"])
    p = onnx_mod.load_policy(out)
    assert p["history"] == history and p["manifest"]["observation_dim"] == 45 * history
    assert all(np.array_equal(x, y) for x, y in zip(p["weights"], w))
    assert all(np.array_equal(x, y) for x, y in zip(p["biases"], b))
    s = verify(out)
    assert s["max_abs_diff"] < 1e-5 and s["fixtures"] == 32
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out / "policy.onnx"), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(1).normal(0.0, 1.0, (5, 45 * history)).astype(np.float32)
    got = np.concatenate([sess.run(None, {"obs": r[None]})[0] for r in x])
    assert np.abs(got - onnx_mod._torch_forward(w, b, x)).max() < 1e-5


def test_from_npz(tmp_path):
    w, b = _layers(45, seed=3)
    src = tmp_path / "policy.npz"
    obs = np.random.default_rng(2).normal(0.0, 0.5, (40, 45)).astype(np.float32)
    np.savez(src, n_layers=4, scale=np.array([0.32, 0.45, 0.42] * 4, dtype=np.float32),
             meta=json.dumps({"omni": True}), fixture_obs=obs, **{f"W{i}": x for i, x in enumerate(w)},
             **{f"b{i}": x for i, x in enumerate(b)})
    onnx_mod.from_npz(src, tmp_path / "pol", {"task": {"name": "walk", "actuator": "bam_t5ub06"}})
    p = onnx_mod.load_policy(tmp_path / "pol")
    assert p["manifest"]["omni"] is True and p["history"] == 1 and p["manifest"]["command"] == {"source": "velocity"}
    assert np.allclose(p["scale"], [0.32, 0.45, 0.42] * 4)
    assert np.array_equal(np.load(tmp_path / "pol" / "fixture.npz")["obs"], obs[:32])
    verify(tmp_path / "pol")


def test_manifest_validation():
    m = manifest_mod.build([0.1, 0.4, 0.42] * 4, omni=False)
    assert m["observation_dim"] == 45 and m["history"] == 1 and m["contract_sha256"] == paths.sha256(paths.CONTRACT_JSON)
    for bad in ({"observation_dim": 90}, {"task": {"source": "/opt/runs/x"}}, {"task": "C:\\runs\\x"},
                {"unknown_key": 1}, {"command": {"source": "joystick"}}, {"robot": "other"}):
        with pytest.raises(ValueError):
            manifest_mod.validate(dict(m, **bad))
    manifest_mod.validate(dict(m, history=2, observation_dim=90))


@pytest.mark.skipif(not paths.POLICIES.is_dir(), reason="no policies/ directory")
@pytest.mark.parametrize("policy", sorted(p.name for p in paths.POLICIES.glob("*") if (p / "policy.onnx").is_file())
                         if paths.POLICIES.is_dir() else [])
def test_shipped_policies_verify(policy):
    verify(paths.POLICIES / policy)
