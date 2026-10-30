"""A training checkpoint as ``assets/policies/<name>/{policy.onnx,fixture.npz,config.json}``.

    python -m mp4_isaac.export runs/<run>/model_1499.pt --name isaac_flat
    python -m mp4_isaac.verify assets/policies/isaac_flat

The scale, history and command source come from the run's ``config.json``. A run with the heading loop is exported
with ``heading.pi_step`` as the command source. The ONNX metadata key is ``mp3_manifest_json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import manifest
from .onnx_policy import actor_layers, write
from .paths import ASSETS, sha256


def task_block(cfg: dict) -> dict:
    o = cfg["task"]["options"]
    return {"name": cfg["task"]["name"], "simulator": cfg["task"]["simulator"], "actuator": cfg["task"]["arm"],
            "terrain": o["terrain"], "heading_law": "pi" if o["heading"] == "loop" else None}


def export(checkpoint, name: str, out: Path | None = None) -> Path:
    ckpt = Path(checkpoint)
    cfg_path = ckpt.parent / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"{cfg_path}: export from the run directory that holds config.json")
    cfg = json.loads(cfg_path.read_text())
    o = cfg["task"]["options"]
    command = manifest.heading_loop_command() if o["heading"] == "loop" else {"source": "velocity"}
    history = int(cfg["variant_settings"]["history"])
    built = manifest.build(
        o["action_scale"], omni=True, history=history, command=command, task=task_block(cfg),
        checkpoint_sha256=sha256(ckpt),
    )
    weights, biases = actor_layers(ckpt)
    out_dir = out or ASSETS / "policies" / name
    path = write(out_dir, weights, biases, built)
    shutil.copyfile(cfg_path, Path(out_dir) / "config.json")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help="runs/<run>/model_<k>.pt (config.json next to it)")
    ap.add_argument("--name", required=True, help="the policy directory: assets/policies/<name>")
    ap.add_argument("--out", type=Path, help="another directory instead of assets/policies/<name>")
    a = ap.parse_args()
    print(export(a.checkpoint, a.name, a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
