# Installing the mp4-sim environment

## Requirements

- [uv](https://docs.astral.sh/uv/) (installs Python 3.12 if it is missing)
- GPU mode: an NVIDIA GPU with a recent driver. The CUDA toolkit (`nvcc`) is
  unnecessary; torch ships its own CUDA runtime.

## Modes

| Mode | For | Installs |
|---|---|---|
| `gpu` | Machines with an NVIDIA GPU (Windows, Linux) | mjlab, MuJoCo Warp, warp-lang, rsl_rl, CUDA 12.8 torch, plus everything in `cpu` |
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
   choosing the torch build with uv's `--torch-backend` (`cu128` or `cpu`).
3. Install mp4-sim itself in editable mode. `pyproject.toml` lists only the
   packages both modes share; backend packages and all pins come from
   `requirements/`.
4. Import every key package, report whether torch sees CUDA, and list the
   onnxruntime execution providers. In `gpu` mode the check fails if CUDA is
   unavailable.

Use `--venv` / `-VenvDir` for a different location. Re-running is safe; the
venv is reused.

## Pins and why

All versions live in `requirements/constraints.txt`.

| Package | Version | Reason |
|---|---|---|
| Python | 3.12 | Required by mjlab-based training and the actuator code |
| `mujoco` | 3.14.0 | Same build used when policies are scored, so training physics match scoring physics |
| `torch` | 2.9.1 | mjlab 1.3.0's ONNX export uses the legacy exporter (`dynamo=False`, opset 18), written for this torch line |
| `mjlab` | 1.3.0 | Training framework and ONNX export |
| `mujoco-warp` | 3.8.1 | Newest version that still has `ls_parallel`, which the actuator bridge uses |
| `warp-lang` | 1.12.0 | `wp.context` was removed in 1.15 |
| `rsl-rl-lib` | 5.0.1 | Pinned by mjlab 1.3.0 |
| `onnx` / `onnxruntime` | 1.23.1 / 1.30.0 | Versions the export path was verified with |

A dry run currently resolves to 134 packages in `gpu` mode and 43 in `cpu`
mode.

## Known risks

- **Actuator models on this MuJoCo line.** The fitted servo models were
  exercised on `mujoco 3.7` and `mujoco-warp 3.7`. Once that code is in the
  tree, check it on 3.14.0 / 3.8.1 before trusting training results.
- **Windows GPU training may need MSVC.** Gait code that compiles to a DLL
  needs `cl.exe`. If that code is added, install Visual Studio's "Desktop
  development with C++" workload. The installer leaves this to you.
- **Headless Linux** needs EGL for rendering: `export MUJOCO_GL=egl`.
- **Weights & Biases** is pulled in by mjlab. Set `WANDB_MODE=disabled` when
  running offline.
