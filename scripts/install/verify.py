"""Post-install check run by install.ps1 / install.sh: no mujoco-core, imports, versions, and the robots' files against
their recorded hashes.

Usage: python verify.py {gpu|cpu}
"""

import importlib
import sys

CPU_MODULES = ["numpy", "scipy", "mujoco", "torch", "mp4_sim"]
GPU_MODULES = CPU_MODULES + ["warp", "mujoco_warp", "mjlab"]


def mujoco_core() -> str | None:
    """The installed mujoco-core version, if any. mujoco-core (required by mjviser 0.0.17 and newer) installs its own
    mujoco/ package over the pinned mujoco and breaks `import mujoco`; the constraints keep it out."""
    from importlib import metadata

    for name in ("mujoco-core", "mujoco_core"):
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return None


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    modules = GPU_MODULES if mode == "gpu" else CPU_MODULES

    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    failed = []
    core = mujoco_core()
    if core:
        failed.append("mujoco-core")
        print(f"  FAIL mujoco-core {core} is installed: it overwrites the pinned mujoco's files. Delete the virtual "
              "environment and run the installer again (requirements/constraints.txt keeps mujoco-core out).")
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

    if "mp4_sim" not in failed:
        from mp4_sim import robots

        for name in robots.ROBOTS:
            try:
                n = len(robots.check_files(name))
                print(f"  ok   robot {name:<14} {n} files match their recorded sha256")
            except Exception as e:
                failed.append(f"robot {name}")
                print(f"  FAIL robot {name:<14} {e}")

    if mode == "gpu" and "mujoco_warp" not in failed and "mp4_sim" not in failed:
        from mp4_sim.envs.patch_warp import is_patched

        state = is_patched()
        print(f"  mujoco_warp patched: {state}  (python -m mp4_sim.envs.patch_warp)")
        if not state:
            failed.append("mujoco_warp patch")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
