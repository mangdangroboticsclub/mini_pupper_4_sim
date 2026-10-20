# mp4-sim

A Python project for simulating the Mini Pupper and training locomotion
policies for it, on any machine. Training ends in one artifact: an ONNX
policy the robot can run.

> **Status:** the Python environment, the robot models (SKU1, SKU2), the
> policy contract, and the BAM servo models are in place. The CPU and GPU
> environments, training, and export code are still to be added.
>
> **The SKU2 model is provisional.** It will be regenerated from the corrected
> SKU2 URDF; see [src/mp4_sim/robot/assets/README.md](src/mp4_sim/robot/assets/README.md).

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
  paths.py          asset locations and the file hash used for provenance
  contract.py       observation / action definition shared with the robot
  robot/
    assets/         MJCF, URDF, meshes, servo maps per SKU (sku_1, sku_2)
    sku.py          SKU registry (MP4_SIM_SKU selects one)
    sku2.py         SKU2 model as trained: weighed masses, IMU sensors, mjlab config
    kinematics.py   leg kinematics
    mapping.py      servo id / direction / zero mapping
    rig.py          ghost rig used when building models
  actuators/
    bam/            vendored BAM servo models (NumPy, MuJoCo, mjlab)
    bam_servo.py    mjlab actuator wrapping BAM, as used on the robot
    arms.py         actuator choices for training (stock, bam_t5ub06)
    params/md01/    fitted servo parameters (t5ub06)
  cpu/              MuJoCo + Gymnasium environments, Stable-Baselines3 training
  gpu/              mjlab tasks, rsl_rl training on MuJoCo Warp
  export/           ONNX export, manifest embedding, verification
cad/                raw CAD (SolidWorks URDF) exports the models are built from
tools/
  build_model.py    regenerate a SKU's MJCF, parts, and servo map from its URDF
  fix_sku2_urdf.py  clean up the raw SKU2 export
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
