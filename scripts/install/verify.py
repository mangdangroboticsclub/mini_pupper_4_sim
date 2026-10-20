"""Post-install check run by install.ps1 / install.sh.

Usage: python verify.py {gpu|cpu}
"""

import importlib
import sys

CPU_MODULES = ["numpy", "scipy", "mujoco", "gymnasium", "stable_baselines3", "torch",
               "onnx", "onnxruntime", "mp4_sim"]
GPU_MODULES = CPU_MODULES + ["warp", "mujoco_warp", "mjlab", "rsl_rl"]


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    modules = GPU_MODULES if mode == "gpu" else CPU_MODULES

    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    failed = []
    for name in modules:
        try:
            module = importlib.import_module(name)
            print(f"  ok   {name:<20} {getattr(module, '__version__', '')}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL {name:<20} {type(e).__name__}: {e}")

    if "torch" not in failed:
        import torch

        print(f"  torch cuda available: {torch.cuda.is_available()}  (build: {torch.version.cuda})")
        if mode == "gpu" and not torch.cuda.is_available():
            failed.append("torch.cuda")

    if "onnxruntime" not in failed:
        import onnxruntime as ort

        print(f"  onnxruntime providers: {ort.get_available_providers()}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
