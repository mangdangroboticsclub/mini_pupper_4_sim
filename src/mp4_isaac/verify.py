"""Check an exported policy.

    python -m mp4_isaac.verify assets/policies/<name>

On onnxruntime: one float32 input ``[1, observation_dim]``, one float32 output ``[1, 12]``, a valid manifest whose
contract hash is this package's ``sku2_contract.json``, and outputs within 1e-5 of ``fixture.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import manifest as manifest_mod
from .paths import contract_json

TOLERANCE = 1e-5


def verify(policy_dir, contract_file=None) -> dict:
    import onnx
    import onnxruntime as ort

    d = Path(policy_dir)
    onnx_path = d / "policy.onnx" if d.is_dir() else d
    d = onnx_path.parent
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    props = {p.key: p.value for p in model.metadata_props}
    if manifest_mod.MANIFEST_KEY not in props:
        raise AssertionError(f"no {manifest_mod.MANIFEST_KEY} in the metadata")
    for key in props:
        if manifest_mod._PATHLIKE.search(key):
            raise AssertionError(key)
    m = manifest_mod.validate(json.loads(props[manifest_mod.MANIFEST_KEY]),
                              contract_file if contract_file is not None else contract_json())
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ins, outs = sess.get_inputs(), sess.get_outputs()
    if len(ins) != 1 or len(outs) != 1:
        raise AssertionError((ins, outs))
    if ins[0].type != "tensor(float)" or outs[0].type != "tensor(float)":
        raise AssertionError((ins[0].type, outs[0].type))
    if list(ins[0].shape) != [1, m["observation_dim"]]:
        raise AssertionError(ins[0].shape)
    if list(outs[0].shape) != [1, m["action_dim"]]:
        raise AssertionError(outs[0].shape)
    fx = np.load(d / "fixture.npz")
    obs, want = fx["obs"].astype(np.float32), fx["actions"].astype(np.float32)
    if obs.ndim != 2 or obs.shape[1] != m["observation_dim"]:
        raise AssertionError(obs.shape)
    got = np.concatenate([sess.run(None, {ins[0].name: o[None]})[0] for o in obs])
    if not np.isfinite(got).all():
        raise AssertionError("non-finite outputs")
    err = float(np.abs(got - want).max())
    if err >= TOLERANCE:
        raise AssertionError(f"max |onnxruntime - fixture| = {err:.3g}")
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
