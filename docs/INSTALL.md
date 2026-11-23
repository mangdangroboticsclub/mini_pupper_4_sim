# Installing the mp4-sim environment

## Requirements

- [uv](https://docs.astral.sh/uv/) 0.12.7 or newer (a release that knows the
  `cu130` torch backend). uv installs Python 3.12 if it is missing.
- GPU mode: an NVIDIA GPU with driver 580 or newer (torch is the CUDA 13.0
  build). The CUDA toolkit (`nvcc`) is unnecessary; torch ships its own CUDA
  runtime.
- CPU mode on macOS: macOS 14 or newer on Apple silicon. torch 2.14 has no
  wheel for Intel Macs.

## Modes

| Mode | For | Installs |
|---|---|---|
| `gpu` | Machines with an NVIDIA GPU (Windows, Linux): the simulation environments (`mp4_sim.envs`) | mjlab, MuJoCo Warp, warp-lang, CUDA 13.0 torch, plus everything in `cpu` |
| `cpu` | Any machine, including macOS and laptops without CUDA: the robot models, calibration, joint limits and servo models on plain MuJoCo | MuJoCo, NumPy, SciPy, CPU torch |

`auto` (the default) picks `gpu` when `nvidia-smi` is on `PATH` (except on
macOS), and `cpu` otherwise, for example on a machine without an NVIDIA driver.

## Running the installer

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1 [-Mode auto|gpu|cpu] [-DryRun] [-SkipVerify]
```

```bash
# Linux / macOS
bash scripts/install/install.sh [--mode auto|gpu|cpu] [--dry-run] [--skip-verify]
```

Both scripts:

1. Create `.venv` in the project root with Python 3.12.
2. Install `requirements/<mode>.txt` under `requirements/constraints.txt`,
   choosing the torch build with uv's `--torch-backend` (`cu130` or `cpu`;
   on macOS, PyPI's torch).
3. Install mp4-sim itself in editable mode. `pyproject.toml` lists only the
   packages both modes share; backend packages and all pins come from
   `requirements/`.
4. In `gpu` mode, patch MuJoCo Warp (`.venv/bin/python -m mp4_sim.envs.patch_warp`, see
   below).
5. Check that `mujoco-core` is not installed (see "Pins and why"), import
   every key package, report whether torch sees CUDA, and check each robot's
   model and measured files against their recorded sha256
   (`mp4_sim.robots.check_files`). In `gpu` mode the check also fails if CUDA
   is unavailable or MuJoCo Warp is not patched.

Use `--venv` / `-VenvDir` for a different location. Re-running is safe; the
venv is reused.

## The MuJoCo Warp patch

When a geom touches more than 50 heightfield cells (a fallen body on fine
terrain), mujoco-warp 3.7.0.1 skips the extra contacts and prints a warning
from the GPU kernel, for every geom pair and every physics substep. With
thousands of robots simulated on rough ground that slows the simulation
several-fold and fills the disk. `.venv/bin/python -m mp4_sim.envs.patch_warp` removes
only the warning; the contacts are handled as before. The feet never reach the
limit.

`mp4_sim.envs.patch_warp.is_patched()` reports the state, and a trainer built
on this package can refuse an unpatched install. Reinstalling mujoco-warp
undoes the patch: run the command again. `--check` only reports.

## Pins and why

All versions live in `requirements/constraints.txt`: it pins every package the
installer installs, on every platform, so it fixes the whole environment. The
compiled robot models
are fingerprinted on this set (`tests/test_models.py`), and the fitted servo
models were identified on the MuJoCo 3.7 line; policies trained on these
environments depend on it too. Change a pin only together with a
re-evaluation. [mini_pupper_4_rl](https://github.com/mangdangroboticsclub/mini_pupper_4_rl)
has its own constraints file: the same pins plus pyserial.

| Package | Version | Reason |
|---|---|---|
| Python | 3.12 | Required by mjlab and the actuator code |
| `mujoco` | 3.7.0 | The MuJoCo line the servo models were fitted on; the compiled-model fingerprints are pinned on it |
| `mujoco-warp` | 3.7.0.1 | Matches `mujoco` 3.7.0; the version `patch_warp` patches |
| `warp-lang` | 1.12.1 | mjlab requires >= 1.12.0 |
| `mjlab` | 1.3.0 | The simulation framework the environments are written for |
| `rsl-rl-lib` | 5.0.1 | Pinned by mjlab 1.3.0 |
| `torch` | 2.14.0 | CUDA 13.0 wheels (`--torch-backend cu130`) |
| `mjviser` | 0.0.16 | mjlab's viewer, installed by mjlab. 0.0.17 and newer require `mujoco-core`, which installs its own `mujoco/` package over `mujoco` 3.7.0 and breaks `import mujoco` |
| `viser` | 1.1.1 | Required by `mjviser` 0.0.16 |
| `trimesh` | 5.2.0 | mjlab requires >= 4.8.3 |

Every other pin (numpy, scipy, tensordict, tensorboard, protobuf, onnx, onnx-ir,
onnxscript, wandb and the packages they depend on; the nvidia-* and triton
packages only with CUDA torch on Linux, colorama only on Windows) is the
version the set resolves with. `scripts/install/verify.py` fails if
`mujoco-core` is installed; delete `.venv` and run the installer again.

## Known risks

- **Headless Linux** needs EGL for rendering: `export MUJOCO_GL=egl`.
- **Installing other packages into `.venv`** without `-c requirements/constraints.txt`
  can upgrade `mjviser` and bring in `mujoco-core` (see "Pins and why").
- **Weights & Biases** is pulled in by mjlab; set `WANDB_MODE=disabled` when
  running offline.
