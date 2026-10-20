"""Checks run on every exported policy.

    python -m mp4_sim.export.verify policies/<name> [policies/<other> ...]

On onnxruntime's ``CPUExecutionProvider``: one float32 input [1, observation_dim] and one float32 output [1, 12]; the
manifest valid (``manifest.validate``, its contract hash that of the shipped ``sku2_contract.json``); the outputs on
``fixture.npz``'s observations within 1e-5 of the recorded actions, and finite; no metadata value that looks like an
absolute path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .. import paths
from . import manifest as manifest_mod

TOLERANCE = 1e-5


def verify(policy_dir, contract_json=paths.CONTRACT_JSON) -> dict:
    """Raise ``AssertionError`` / ``ValueError`` on the first failed check; return a summary."""
    import onnx
    import onnxruntime as ort

    d = Path(policy_dir)
    onnx_path = d / "policy.onnx" if d.is_dir() else d
    d = onnx_path.parent
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    props = {p.key: p.value for p in model.metadata_props}
    assert manifest_mod.MANIFEST_KEY in props, f"no {manifest_mod.MANIFEST_KEY} in the metadata"
    for key, value in props.items():
        assert not manifest_mod._PATHLIKE.search(key), key
    m = manifest_mod.validate(json.loads(props[manifest_mod.MANIFEST_KEY]), contract_json=contract_json)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    assert len(ins) == 1 and len(outs) == 1, (ins, outs)
    assert ins[0].type == "tensor(float)" and outs[0].type == "tensor(float)", (ins[0].type, outs[0].type)
    assert list(ins[0].shape) == [1, m["observation_dim"]], ins[0].shape
    assert list(outs[0].shape) == [1, m["action_dim"]], outs[0].shape
    fx = np.load(d / "fixture.npz")
    obs, want = fx["obs"].astype(np.float32), fx["actions"].astype(np.float32)
    assert obs.ndim == 2 and obs.shape[1] == m["observation_dim"], obs.shape
    got = np.concatenate([sess.run(None, {ins[0].name: o[None]})[0] for o in obs])
    assert np.isfinite(got).all(), "non-finite outputs"
    err = float(np.abs(got - want).max())
    assert err < TOLERANCE, f"max |onnxruntime - fixture| = {err:.3g}"
    return {"policy": d.name, "observation_dim": m["observation_dim"], "history": m.get("history", 1),
            "fixtures": len(obs), "max_abs_diff": err}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policies", nargs="+", help="policy directories (policy.onnx, fixture.npz)")
    a = ap.parse_args()
    failed = 0
    for p in a.policies:
        try:
            s = verify(p)
            print(f"ok    {s['policy']}: obs {s['observation_dim']} (history {s['history']}), "
                  f"{s['fixtures']} fixtures, max diff {s['max_abs_diff']:.2e}")
        except (AssertionError, ValueError, OSError) as e:
            failed += 1
            print(f"FAIL  {Path(p).name}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
