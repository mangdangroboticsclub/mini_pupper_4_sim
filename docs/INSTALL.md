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
| `gpu` | Machines with an NVIDIA GPU (Windows, Linux) | mjlab, MuJoCo Warp, warp-lang, rsl_rl, CUDA 13.0 torch, plus everything in `cpu` |
| `cpu` | Any machine, including macOS and laptops without CUDA | MuJoCo, Gymnasium, Stable-Baselines3, CPU torch, ONNX tooling |

`auto` (the default) picks `gpu` when `nvidia-smi` is on `PATH`, and picks
`cpu` on macOS.

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
4. In `gpu` mode, patch MuJoCo Warp (`python -m mp4_sim.gpu.patch_warp`, see
   below).
5. Import every key package, report whether torch sees CUDA, and list the
   onnxruntime execution providers. In `gpu` mode the check fails if CUDA is
   unavailable or MuJoCo Warp is not patched.

Use `--venv` / `-VenvDir` for a different location. Re-running is safe; the
venv is reused.

## The MuJoCo Warp patch

When a geom touches more than 50 heightfield cells (a fallen body on fine
terrain), mujoco-warp 3.7.0.1 skips the extra contacts and prints a warning
from the GPU kernel, for every geom pair and every physics substep. With
thousands of robots training on rough ground that slows training several-fold
and fills the disk. `python -m mp4_sim.gpu.patch_warp` removes only the
warning; the contacts are handled as before. The feet never reach the limit.

`mp4_sim.gpu.train` refuses to start on an unpatched install
(`--allow-unpatched` overrides). Reinstalling mujoco-warp undoes the patch:
run the command again. `--check` only reports.

## Pins and why

All versions live in `requirements/constraints.txt` (gymnasium and
stable-baselines3 as minimum versions, the rest pinned). The shipped policies
were trained and evaluated on this set, and the fitted servo models were
identified on the MuJoCo 3.7 line. Change a pin only together with a
re-evaluation.

| Package | Version | Reason |
|---|---|---|
| Python | 3.12 | Required by mjlab-based training and the actuator code |
| `mujoco` | 3.7.0 | The MuJoCo line the servo models were fitted on and the policies trained on |
| `mujoco-warp` | 3.7.0.1 | Matches `mujoco` 3.7.0; the version `patch_warp` patches |
| `warp-lang` | 1.12.1 | Version the policies were trained with; mjlab requires >= 1.12.0 |
| `mjlab` | 1.3.0 | Training framework |
| `rsl-rl-lib` | 5.0.1 | Pinned by mjlab 1.3.0; `gpu/algorithms.py` extends its PPO |
| `torch` / `torchvision` | 2.14.0 / 0.29.0 | Version the policies were trained with; CUDA 13.0 wheels (`--torch-backend cu130`) |
| `onnx` / `onnxruntime` | 1.23.0 / 1.30.0 | Versions the export and `mp4_sim.export.verify` were run with |
| `trimesh` | 5.1.0 | Used by `tools/build_model.py` and `tools/fix_sku2_urdf.py`; mjlab requires >= 4.8.3 |

The remaining pins (numpy, scipy, tensordict, tensorboard, protobuf, onnx-ir,
onnxscript) are the versions the set resolves with.

## Known risks

- **Headless Linux** needs EGL for rendering (videos from
  `mp4_sim.gpu.evaluate --video`): `export MUJOCO_GL=egl`.
- **Weights & Biases** is pulled in by mjlab. Training logs to TensorBoard by
  default; set `WANDB_MODE=disabled` when running offline.
