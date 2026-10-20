"""Silence mujoco_warp's heightfield-overflow warning in the environment it runs from (idempotent; ``--check`` only
reports).

    python -m mp4_sim.gpu.patch_warp [--check]

When a geom touches more than 50 heightfield cells (a fallen body on fine terrain), mujoco_warp 3.7.0.1 skips the rest
(``continue``) and prints a warning from the GPU kernel, for every geom pair and every physics substep. In training with
thousands of falling robots that is millions of lines: it slows the run several-fold and fills the disk. The patch
removes only the warning; the skip, and so the physics, is unchanged. The feet never reach the cap (a 4.9 mm sphere on
5 mm cells). The install scripts run it; re-run it after reinstalling mujoco_warp. ``is_patched()`` is the check
``mp4_sim.gpu.train`` runs before training.

The file is replaced (a new file, then ``os.replace``), never written in place: uv hardlinks installed packages to its
cache, and an in-place write would change the cached copy too.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from pathlib import Path

OLD = '''          if count >= MJ_MAXCONPAIR:
            wp.printf(
              "height field collision overflow, number of collisions >= %u - please adjust resolution: \\n decrease the number of hfield rows/cols or modify size of colliding geom\\n",
              MJ_MAXCONPAIR,
            )
            continue'''
NEW = '''          if count >= MJ_MAXCONPAIR:
            # (warning print removed; the contact is still skipped)
            continue'''
# the patched block, whatever its comment says
PATCHED = re.compile(r"if count >= MJ_MAXCONPAIR:\n[ ]*#[^\n]*\n[ ]*continue")


def target() -> Path:
    import importlib.util
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("mujoco_warp is not installed")
    return Path(spec.origin).parent / "_src" / "collision_convex.py"


def is_patched(src: str | None = None) -> bool | None:
    """True when the warning print is removed, False when the original block is there, None when neither is found
    (another mujoco_warp version)."""
    src = target().read_text() if src is None else src
    if OLD in src:
        return False
    return True if PATCHED.search(src) else None


def patch(path: Path, src: str) -> None:
    """Write the patched source to a new file next to ``path`` and move it over ``path``."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(src.replace(OLD, NEW))
        os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report; exit 1 when not patched")
    a = ap.parse_args()
    path = target()
    src = path.read_text(encoding="utf-8")
    print(f"{path.name} (mujoco_warp), sha256 {hashlib.sha256(src.encode()).hexdigest()[:16]}")
    state = is_patched(src)
    if state:
        print("patched")
        return 0
    if state is None:
        print("the expected block is not there (another mujoco_warp version?): not patched")
        return 1
    if a.check:
        print("not patched")
        return 1
    patch(path, src)
    print("patched now, sha256", hashlib.sha256(path.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())
