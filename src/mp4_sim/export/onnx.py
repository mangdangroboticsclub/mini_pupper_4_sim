"""Actor-to-ONNX conversion: the actor's mean net as an ONNX graph with the manifest in its metadata.

    python -m mp4_sim.export.onnx --checkpoint runs/<run>/model_<k>.pt --out policies/<name>

``out/policy.onnx``: input ``obs`` float32 [1, D], output ``actions`` float32 [1, 12] (the unclipped action mean),
``Gemm`` layers with ``Elu`` (alpha 1) between them, opset 18, no normalisation; the manifest (``manifest.build``) as
JSON under the metadata key ``mp3_manifest_json``. ``out/fixture.npz``: 32 seeded observations (``obs``) and the
actor's outputs for them (``actions``), which ``mp4_sim.export.verify`` compares against onnxruntime.

A checkpoint's action scale, history and command source come from its run's ``config.json``. ``from_npz`` converts a
policy exported as numpy arrays (``W0, b0, ...``, ``scale``, ``meta``, ``fixture_obs``).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .. import contract
from . import manifest as manifest_mod

OPSET = 18
FIXTURES = 32


def build_model(weights: list, biases: list, manifest: dict):
    """The ONNX model of an ELU MLP (``weights[i]``: (out, in) as torch stores them)."""
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
    graph = helper.make_graph(nodes, "policy", [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, n_in])],
                              [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, n_out])], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)], producer_name="mp4_sim")
    model.ir_version = 10
    helper.set_model_props(model, {manifest_mod.MANIFEST_KEY: json.dumps(manifest, sort_keys=True)})
    onnx.checker.check_model(model)
    return model


def forward(weights: list, biases: list, x) -> np.ndarray:
    """The MLP in numpy, float32 (ELU between the layers)."""
    x = np.asarray(x, dtype=np.float32)
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = x @ np.asarray(w, dtype=np.float32).T + np.asarray(b, dtype=np.float32)
        if i < len(weights) - 1:
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))).astype(np.float32)
    return x


def fixture_observations(history: int, n: int = FIXTURES, seed: int = 0) -> np.ndarray:
    """``n`` seeded observations shaped like training's: small body rates, gravity near upright, a walking command,
    positions near the stand, joint rates, actions within [-1, 1]; each frame of a history drawn on its own."""
    rng = np.random.default_rng(seed)
    h = max(int(history), 1)
    g = rng.normal(0.0, 0.1, (n, h, 3)) + np.array([0.0, 0.0, -1.0])
    parts = {"ang_vel": rng.normal(0.0, 0.5, (n, h, 3)), "gravity": g / np.linalg.norm(g, axis=-1, keepdims=True),
             "command": np.stack([rng.uniform(0.0, 0.15, (n, h)), np.zeros((n, h)), rng.uniform(-0.4, 0.4, (n, h))],
                                 -1),
             "pos": rng.normal(0.0, 0.15, (n, h, 12)), "vel": rng.normal(0.0, 1.0, (n, h, 12)),
             "prev_action": rng.uniform(-1.0, 1.0, (n, h, 12))}
    return np.concatenate([parts[name].reshape(n, -1) for name, _ in contract.OBS_TERMS], -1).astype(np.float32)


def write(out_dir, weights: list, biases: list, manifest: dict, fixture_obs=None, fixture_actions=None) -> Path:
    """``policy.onnx`` and ``fixture.npz`` in ``out_dir``; the fixture's actions default to the numpy forward pass."""
    import onnx

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if weights[0].shape[1] != manifest["observation_dim"]:
        raise ValueError(f"input width {weights[0].shape[1]} != observation_dim {manifest['observation_dim']}")
    obs = fixture_observations(manifest.get("history", 1)) if fixture_obs is None else np.asarray(fixture_obs,
                                                                                                    np.float32)
    act = forward(weights, biases, obs) if fixture_actions is None else np.asarray(fixture_actions, np.float32)
    onnx.save(build_model(weights, biases, manifest), out / "policy.onnx")
    np.savez(out / "fixture.npz", obs=obs, actions=act)
    return out / "policy.onnx"


def load_policy(path) -> dict:
    """An exported ``policy.onnx`` (or its directory): ``weights`` and ``biases`` (lists, torch layout), ``scale``
    (12, float32), ``history`` and the ``manifest``."""
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
    """The actor's mean net (``mlp.<k>.weight`` / ``bias``) from an rsl_rl checkpoint."""
    import torch

    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = ck["actor_state_dict"]
    idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
    return ([sd[f"mlp.{i}.weight"].numpy().astype(np.float32) for i in idx],
            [sd[f"mlp.{i}.bias"].numpy().astype(np.float32) for i in idx])


def _torch_forward(weights, biases, x) -> np.ndarray:
    """The actor's forward pass in torch (the training-side reference for the fixture)."""
    import torch

    y = torch.from_numpy(np.asarray(x, dtype=np.float32))
    with torch.no_grad():
        for i, (w, b) in enumerate(zip(weights, biases)):
            y = torch.nn.functional.linear(y, torch.from_numpy(w), torch.from_numpy(b))
            if i < len(weights) - 1:
                y = torch.nn.functional.elu(y)
    return y.numpy()


def run_settings(checkpoint) -> dict:
    """What a terrain run's ``config.json`` (next to its checkpoint) says about the policy: scale, history, command."""
    cfg = json.loads((Path(checkpoint).parent / "config.json").read_text())
    o = cfg["task"]["options"]
    loop = o.get("heading") == "loop"
    command = None
    if loop:
        command = manifest_mod.heading_loop_command(o.get("heading_wz", 0.4) if o.get("heading_law") == "pi" else 0.4)
    task = {"name": "terrain" if cfg.get("terrain_task") else "walk", "actuator": cfg["task"]["arm"],
            "terrain": o.get("terrain"), "heading_law": o.get("heading_law") if loop else None}
    return {"scale": o["action_scale"], "history": int(cfg["variant_settings"]["history"]), "omni": True,
            "command": command, "task": task}


def export(source, out_dir, scale=None, *, history: int | None = None, omni: bool | None = None,
           command: dict | None = None, task: dict | None = None, manifest_overrides: dict | None = None) -> Path:
    """An rsl_rl checkpoint (``.pt``) or a numpy-array policy (``.npz``) as ``out_dir/policy.onnx`` and
    ``fixture.npz``. Unset options come from the checkpoint's run config (or the npz's own fields)."""
    src = Path(source)
    if src.suffix == ".npz":
        return from_npz(src, out_dir, dict(manifest_overrides or {}, **{k: v for k, v in (
            ("action_scale_rad", scale), ("history", history), ("omni", omni), ("command", command), ("task", task))
            if v is not None}))
    run = run_settings(src) if (src.parent / "config.json").is_file() else {}
    scale = run.get("scale") if scale is None else scale
    if scale is None:
        raise ValueError("no action scale: no config.json next to the checkpoint; export from the run directory that holds "
                         "config.json, or call export(..., scale=...) from Python")
    history = run.get("history", 0) if history is None else history
    omni = run.get("omni", True) if omni is None else omni
    weights, biases = actor_layers(src)
    from .. import paths
    m = manifest_mod.build(scale, omni=omni, history=history, command=command or run.get("command"),
                           task=task or run.get("task"), checkpoint_sha256=paths.sha256(src))
    if manifest_overrides:
        m.update(manifest_overrides)
        manifest_mod.validate(m)
    obs = fixture_observations(m["history"])
    return write(out_dir, weights, biases, m, obs, _torch_forward(weights, biases, obs))


def from_npz(npz_path, out_dir, manifest_overrides: dict | None = None) -> Path:
    """A policy saved as numpy arrays (``n_layers``, ``W<i>``, ``b<i>``, ``scale``, ``meta`` JSON with ``omni`` and
    ``history``, ``fixture_obs``) as ``out_dir/policy.onnx``; the fixture is the first 32 of its observations."""
    d = np.load(npz_path, allow_pickle=False)
    n = int(d["n_layers"])
    weights = [d[f"W{i}"].astype(np.float32) for i in range(n)]
    biases = [d[f"b{i}"].astype(np.float32) for i in range(n)]
    meta = json.loads(str(d["meta"])) if "meta" in d.files else {}
    o = dict(manifest_overrides or {})
    scale = o.pop("action_scale_rad", d["scale"].tolist())
    m = manifest_mod.build(scale, omni=bool(o.pop("omni", meta.get("omni", False))),
                           history=int(o.pop("history", meta.get("history", 0) or 0)), command=o.pop("command", None),
                           task=o.pop("task", None), **o)
    obs = d["fixture_obs"][:FIXTURES] if "fixture_obs" in d.files else fixture_observations(m["history"])
    return write(out_dir, weights, biases, m, obs, _torch_forward(weights, biases, obs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="an rsl_rl checkpoint (runs/<run>/model_<k>.pt; config.json next to it)")
    src.add_argument("--npz", help="a policy saved as numpy arrays (W0, b0, ..., scale, meta)")
    ap.add_argument("--out", required=True, help="the policy directory to write (policies/<name>)")
    ap.add_argument("--command", choices=manifest_mod.COMMAND_SOURCES,
                    help="the command source (default: the run's; heading_loop: the robot's PI loop)")
    ap.add_argument("--heading-wz", type=float, default=0.4, help="heading_loop: the loop's output limit (rad/s)")
    ap.add_argument("--omni", action=argparse.BooleanOptionalAction, default=None,
                    help="the policy observes the whole command (vx, vy, wz)")
    a = ap.parse_args()
    command = None
    if a.command == "heading_loop":
        command = manifest_mod.heading_loop_command(a.heading_wz)
    elif a.command == "velocity":
        command = {"source": "velocity"}
    path = export(a.checkpoint or a.npz, a.out, command=command, omni=a.omni)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
