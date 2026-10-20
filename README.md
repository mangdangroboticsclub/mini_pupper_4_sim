# mp4-sim

A Python project for simulating the Mini Pupper and training locomotion
policies for it, on any machine. Training ends in one artifact: an ONNX
policy the robot can run.

> **Status:** scaffold. The Python environment (`requirements/`,
> `scripts/install/`), the package layout under `src/mp4_sim/`, and these docs
> are in place. Each module describes what it will contain; the simulator,
> training, and export code is still to be added.

## What it is

One codebase and one environment for the whole pipeline:

```
train (GPU or CPU)  ->  checkpoint  ->  ONNX + manifest  ->  robot
```

- **Two training backends, one robot.** GPU-parallel training with mjlab and
  MuJoCo Warp on NVIDIA machines; MuJoCo and Gymnasium on CPU-only machines.
  Both use the same robot model, actuator models, and observation/action
  definition.
- **One export path.** Every trained policy, from either backend, is exported
  to ONNX with an embedded manifest. See
  [docs/POLICY_CONTRACT.md](docs/POLICY_CONTRACT.md).
- **One environment.** A single pinned virtual environment, installed by one
  script, on Windows, Linux, or macOS (the CPU environment on macOS).

Design notes are in [docs/RATIONALE.md](docs/RATIONALE.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/). The script installs Python 3.12 if needed.

```powershell
# Windows (auto-detects GPU vs CPU)
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1
```

```bash
# Linux / macOS
bash scripts/install/install.sh
```

Pass `-Mode cpu` / `--mode cpu` to force the CPU-only environment, or
`-DryRun` / `--dry-run` to see what would be installed. Details, pins, and the
reasons behind them are in [docs/INSTALL.md](docs/INSTALL.md).

## Layout

```
src/mp4_sim/
  contract.py       observation / action / manifest definition (single source)
  robot/            robot models per SKU (assets/sku_1, assets/sku_2) and servo maps
  actuators/        fitted servo models (params/) with NumPy and mjlab bindings
  cpu/              MuJoCo + Gymnasium environments, Stable-Baselines3 training
  gpu/              mjlab tasks, rsl_rl training on MuJoCo Warp
  export/           ONNX export, manifest embedding, verification
tests/
requirements/
  constraints.txt   single pin set shared by both modes
  gpu.txt           mjlab / MuJoCo Warp / CUDA torch + everything in cpu.txt
  cpu.txt           MuJoCo / Gymnasium / Stable-Baselines3 / CPU torch
scripts/install/
  install.ps1       Windows installer
  install.sh        Linux / macOS installer
  verify.py         post-install import / CUDA check used by both installers
docs/
  RATIONALE.md      why the stack is shaped this way
  INSTALL.md        environment modes, pins, and known risks
  POLICY_CONTRACT.md  the ONNX interface every export must satisfy
```
