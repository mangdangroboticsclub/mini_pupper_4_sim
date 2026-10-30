"""The actor's mean net as an ONNX graph, with the manifest in its metadata.

``policy.onnx``: input ``obs`` float32 ``[1, D]``, output ``actions`` float32 ``[1, 12]``, Gemm layers with Elu
between them, opset 18. ``fixture.npz`` holds 32 observations and the actor's outputs for them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import contract
from . import manifest as manifest_mod

OPSET = 18
FIXTURES = 32


def build_model(weights: list, biases: list, manifest: dict):
    """An ELU MLP. ``weights[i]`` is ``(out, in)``, the layout torch stores."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    n_in, n_out = weights[0].shape[1], weights[-1].shape[0]
    nodes, inits, x = [], [], "obs"
    for i, (w, b) in enumerate(zip(weights, biases)):
        inits += [numpy_helper.from_array(np.asarray(w, dtype=np.float32), f"W{i}"),
                  numpy_helper.from_array(np.asarray(b, dtype=np.float32), f"b{i}")]
        last = i == len(weights) - 1
        y = "actions" if last else f"h{i}"
        nodes.append(helper.make_node("Gemm", [x, f"W{i}", f"b{i}"], [y if last else f"z{i}"], transB=1,
                                      name=f"gemm{i}"))
        if not last:
            nodes.append(helper.make_node("Elu", [f"z{i}"], [y], alpha=1.0, name=f"elu{i}"))
        x = y
    graph = helper.make_graph(
        nodes, "policy",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_in])],
        [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n_out])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="mp4_isaac")
    model.ir_version = 10
    helper.set_model_props(model, {manifest_mod.MANIFEST_KEY: json.dumps(manifest, sort_keys=True)})
    onnx.checker.check_model(model)
    return model


def forward(weights: list, biases: list, x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = x @ np.asarray(w, dtype=np.float32).T + np.asarray(b, dtype=np.float32)
        if i < len(weights) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))).astype(np.float32)
    return x


def fixture_observations(history: int, n: int = FIXTURES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = max(int(history), 1)
    g = rng.normal(0.0, 0.1, (n, h, 3)) + np.array([0.0, 0.0, -1.0])
    parts = {
        "ang_vel": rng.normal(0.0, 0.5, (n, h, 3)),
        "gravity": g / np.linalg.norm(g, axis=-1, keepdims=True),
        "command": np.stack([rng.uniform(0.0, 0.15, (n, h)), np.zeros((n, h)), rng.uniform(-0.4, 0.4, (n, h))], -1),
        "pos": rng.normal(0.0, 0.15, (n, h, 12)),
        "vel": rng.normal(0.0, 1.0, (n, h, 12)),
        "prev_action": rng.uniform(-1.0, 1.0, (n, h, 12)),
    }
    return np.concatenate([parts[name].reshape(n, -1) for name, _ in contract.OBS_TERMS], -1).astype(np.float32)


def write(out_dir, weights: list, biases: list, manifest: dict, fixture_obs=None, fixture_actions=None) -> Path:
    import onnx

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if weights[0].shape[1] != manifest["observation_dim"]:
        raise ValueError(f"input width {weights[0].shape[1]} != observation_dim {manifest['observation_dim']}")
    obs = fixture_observations(manifest.get("history", 1)) if fixture_obs is None else np.asarray(fixture_obs, np.float32)
    act = forward(weights, biases, obs) if fixture_actions is None else np.asarray(fixture_actions, np.float32)
    onnx.save(build_model(weights, biases, manifest), out / "policy.onnx")
    np.savez(out / "fixture.npz", obs=obs, actions=act)
    return out / "policy.onnx"


def load_policy(path) -> dict:
    """An exported ``policy.onnx`` or its directory: weights, biases, scale, history, manifest."""
    import onnx
    from onnx import numpy_helper

    p = Path(path)
    p = p / "policy.onnx" if p.is_dir() else p
    model = onnx.load(str(p))
    props = {q.key: q.value for q in model.metadata_props}
    if manifest_mod.MANIFEST_KEY not in props:
        raise ValueError(f"{p.name}: no {manifest_mod.MANIFEST_KEY} metadata")
    m = json.loads(props[manifest_mod.MANIFEST_KEY])
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    weights, biases = [], []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            w = init[node.input[1]]
            weights.append((w if trans_b else w.T).astype(np.float32))
            biases.append(init[node.input[2]].astype(np.float32))
    return {"weights": weights, "biases": biases, "scale": np.asarray(m["action_scale_rad"], dtype=np.float32),
            "history": int(m.get("history", 1)), "manifest": m}


def actor_layers(checkpoint) -> tuple[list, list]:
    """The actor mean net (``mlp.<k>.weight`` / ``bias``) from an rsl_rl checkpoint."""
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    return ([sd[f"mlp.{i}.weight"].numpy().astype(np.float32) for i in idx],
            [sd[f"mlp.{i}.bias"].numpy().astype(np.float32) for i in idx])
