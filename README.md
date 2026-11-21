# Mini Pupper v4 Simulator

The simulation foundation for the Mini Pupper v4 quadruped: robot models of the SKU2 and the SKU1 with the data
measured on the robots, the servo calibration and joint limits, fitted models of the MD-300-150 servo module, and
simulation environments for [mjlab](https://github.com/mujocolab/mjlab) on MuJoCo Warp (scene, sensors, terrain, plant
randomisation).

## The robots

| | SKU2 | SKU1 |
|---|---|---|
| Body | long | short |
| Electronics | Raspberry Pi 5 + ESP32 | ESP32 |
| Weighed mass | 630.5 g | 413.4 g |
| Calibration | `sku2_contract.json` | `sku1_contract.json` |
| Joint limits | traced by hand on the robot (`sku2_joint_limits.json`, `sku2_hip_envelope.json`) | traced by hand on the robot (`sku1_joint_limits.json`, `sku1_hip_envelope.json`) |

## Requirements

- [uv](https://docs.astral.sh/uv/) 0.12.7 or newer. The installer fetches Python 3.12 if needed.
- The simulation environments (`mp4_sim.envs`): Linux or Windows with an NVIDIA GPU and driver 580 or newer.
- CPU mode (the robot models, calibration, joint limits and servo models on plain MuJoCo, without CUDA): Linux,
  Windows, or macOS 14+ on Apple silicon.

## Install

```bash
git clone https://github.com/mangdangroboticsclub/mini_pupper_4_sim.git && cd mini_pupper_4_sim
bash scripts/install/install.sh                    # GPU or CPU mode, picked automatically (--mode cpu forces CPU)
.venv/bin/python -m pytest tests                   # GPU tests are skipped without CUDA
```

On Windows: `powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1`. Installer options and pins are in
[docs/INSTALL.md](docs/INSTALL.md). Run the commands without an inherited `PYTHONPATH` (prefix them with
`env -u PYTHONPATH` if your shell sets one).

## Use

```python
from mp4_sim import robots
from mp4_sim.calibration import Contract
from mp4_sim.robot import model

robot = robots.get("sku1")                              # or "sku2"
spec = model.get_spec(legs=model.Legs(robot="sku1"))    # MjSpec: weighed masses, IMU, sphere feet
m = spec.compile()
d = model.stand_data(m, robot)                          # the calibrated stand pose
c = Contract.load(robot=robot)                          # servo signs, stand, firmware units

from mp4_sim.envs.scene import SceneOptions, scene_cfg  # GPU install
scene = scene_cfg(SceneOptions(robot="sku1", terrain="rough", num_envs=1024))
```

Nothing in the package keeps a current robot: pass it to each function (left out, it is the SKU2). `scene_cfg` returns the mjlab scene and the
plant's randomisation events; a task adds its observations, actions, rewards and terminations on top.

## The RL skills built on it

[mini_pupper_4_rl](https://github.com/mangdang-bam/mini_pupper_4_rl) trains and evaluates a walker and a get-up policy
on these environments, with `--robot sku2|sku1`, and runs the SKU2 policies on the robot. It installs this package at a
tagged release and checks the hashes of the files its policies were exported for.

## What is where

```
src/mp4_sim/
  calibration.py    servo calibration: stand, signs, offsets, firmware units (numpy only, no other mp4_sim import)
  limits.py         measured joint limits: loading, joint ranges, applying them to a model
  robots.py         the robots (sku2, sku1): their files and recorded sha256
  robot/            the robot model (model.py), firmware kinematics, servo maps; assets/ per SKU: MJCF, meshes,
                    URDF, servo map, measured data (see robot/assets/README.md)
  actuators/        BAM (Better Actuator Models), vendored, with the MD-300-150 fits; the GPU servo (mjlab) and the
                    CPU servo bank (plain MuJoCo)
  envs/             scene_cfg (robot, sensors, terrain, randomisation events), terrain, spawns on terrain, the hood
                    attachment, centre-of-mass height, contact edits for falls, measured sensor noise, a MuJoCo Warp
                    patch; data/: measured staircases and the height map of a 3D-printed test tile
scripts/install/    installers (Linux/macOS, Windows) and the post-install check
cad/                the SKU2's URDF and meshes as exported from CAD (reference; the package doesn't read them)
docs/               INSTALL, ROBOTS
tests/              model and data hashes, compiled-model fingerprints, calibration, limits, servo models, terrain,
                    GPU scene smoke tests, wheel contents
```
