# Implementation

Follows [DESIGN.md](DESIGN.md). Do the staging check before installing Isaac
Lab. Conversion is the step that needs the NVIDIA packages.

## 1. Environment

Python 3.12 and `uv`. The virtualenv defaults to `.venv` in this repository.

```bash
bash scripts/install.sh
bash scripts/install.sh /path/to/venv
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install.ps1
powershell -ExecutionPolicy Bypass -File scripts\install.ps1 -Venv D:\venvs\mp4-isaac
```

`MP4_ISAAC_VENV` selects the directory when you do not pass one. The bash script activates `bin/python`. The PowerShell script installs into `Scripts\python.exe` and does not need the virtualenv to be activated. Isaac Lab 3 lists Windows 11 for that path.

The script:

1. Installs `uv` for the current user if it is missing.
2. Creates the virtualenv with CPython 3.12.
3. Installs `isaaclab[importers]==3.0.0` from `pypi.nvidia.com`, with the
   Isaac Lab 3.0 override file so the importer extra does not downgrade the
   rest of the stack. The command is the one in the Isaac Lab 3.0 install
   guide (`--torch-backend cu130`).
4. Replaces that torch with `2.11.0+cu128` and `torchvision==0.26.0+cu128`.
   The published wheel's CUDA 13 build does not load on a CUDA 12.9 driver.
   Re-running the script keeps the CUDA 12.8 build, because that install comes
   last.
5. Installs this package in editable mode.

The script does not request the `isaacsim` extra, so Kit should not start.

## 2. Stage the URDF

From `mp4-isaac/`, with or without the virtualenv:

```bash
PYTHONPATH=src python3 -m mp4_isaac.stage
```

`stage.py` reads

- `assets/sku_2/urdf/sku_2_urdf_fixed.urdf`
- `assets/sku_2/display/display.json`

and writes `.cache/urdf/sku_2_with_display.urdf` with the robot name `sku_2`
(the importer uses that name for the USD directory). It then resolves every
`package://sku_2_urdf_fixed/...` path against the `sku_2` directory and exits
non-zero if a mesh is missing.

Expected staged counts: 15 links (13 CAD + bezel + panel), 12 continuous
joints, 2 fixed joints.

## 3. Convert

With the virtualenv active:

```bash
python -m mp4_isaac.convert
```

This calls the stager, then `isaaclab.sim.converters.UrdfConverter` with the
settings in the design doc. Output goes to `.cache/usd/`. Isaac Lab 3 puts the
interface USD in a subdirectory named from the robot and may add `_1`, `_2`, …
when a previous output is present. The script prints the `usd_path` the
converter actually wrote. `python -m mp4_isaac.convert --force` asks for a
fresh conversion.

`convert.py` does not construct a `SimulationApp`. The standalone importer is
what should run.

## 4. Inspect

```bash
python -m mp4_isaac.inspect
```

Always prints the staged URDF summary. When `.cache/usd` contains a USD layer
and `pxr` imports (it arrives with the importer), the script opens the newest
interface layer and prints:

- the default prim
- links that carry a rigid body or an articulation
- joints and their types
- how many meshes were authored

After a merged conversion the expectation is 13 links and 12 revolute joints,
with the bezel and panel meshes under the base. A count of 15 links means the
fixed screen joints were kept; the asset is still usable, and the screen is a
child of the base rather than part of it.

Checked on this machine after `python -m mp4_isaac.convert`:

- interface USD: `.cache/usd/sku_2_with_display/sku_2_with_display.usda`,
  physics variant `physics`
- 13 rigid bodies, 12 revolute joints, CAD mass 0.244610 kg
- 28 meshes (13 visuals, 13 convex-hull collisions, 2 screen visuals), 271412 faces
- the screen prims are instanceable children of `base_link` and have no collision
- Warp 1.17 sees `cuda:0` (RTX 2060, CUDA toolkit 12.9). PyTorch is
  `2.11.0+cu128`; `torch.cuda.is_available()` is true and a tensor on
  `cuda:0` allocates.

## 5. Files

| Module | Responsibility |
|---|---|
| `paths.py` | Repo root, the SKU2 asset directory, cache paths, the ROS package name `sku_2_urdf_fixed`. |
| `stage.py` | XML edit: materials, display links, mesh resolution. No Isaac import. |
| `convert.py` | Build `UrdfConverterCfg` and run it. |
| `inspect.py` | URDF summary plus an optional `pxr` walk. |
| `drop.py` | Spawn the USD in Newton, drop it onto a floor, and check the articulation. |

`pyproject.toml` does not depend on `isaaclab`. The NVIDIA index and the
override file are install concerns, handled by `scripts/install.sh`.

## 6. Drop test

```bash
python -m mp4_isaac.drop
python -m mp4_isaac.drop --video --seconds 2
```

`--video` writes `.cache/drop.mp4` and does not open a window. This shell has no
display, so `--viz newton_gl` fails in pyglet before a window exists. The clip
is 1280×720 at 30 fps and draws the CAD meshes. Collision stays on the convex
hulls, which are not shown.

Newton MJWarp, one robot, CUDA. The base starts at 0.20 m with every joint at
the CAD zero. Implicit PD holds that pose (`stiffness` 0.2, `damping` 0.02,
`armature` 1e-4). Checked on this machine for 0.5 s:

- 13 bodies and the 12 CAD joints. The screen is not a rigid body.
- Free fall until about 0.16 s, then contact. Joints stay at 0 until the feet land.
- Settled base height 75 mm, tilt under 0.2°, joint motion 0.14 rad where the
  soft hold gives under the weight, joint speed 0.03 rad/s.
- A stiffer hold with no armature (the first attempt, 4 N m/rad) went non-finite
  at 75 ms, before contact. The inertias in the URDF are the plastic CAD values.

## 7. SKU1

```bash
python -m mp4_isaac.convert --sku sku_1
python -m mp4_isaac.drop --sku sku_1 --video .cache/sku_1.mp4 --seconds 0.4
```

SKU1 uses the same part colors as SKU2: yellow body, black frame, black motor,
metal flange, yellow bracket, dark legs, yellow screen. Its URDF still names
one joint `back_riht_leg_upper_joint`.

## 8. flat_omni

```bash
python -m mp4_isaac.play --video --vx 0.15 --seconds 3
```

This runs `assets/policies/flat_omni`. Servo angles are mapped onto the CAD
hinges, and the clip is `.cache/flat_omni.mp4`. A 0.15 m/s forward command stayed upright and covered
0.29 m in 3 s. The hold is a plain PD (`--stiffness 1.5 --armature 5e-4`).
The MuJoCo task's stiffer servo flips this model.

## 9. After this pass

1. Open the USD in a kit-less viewer and look at the assembly pose: thighs
   about −45°, shins about 45°, screen on the front of the base.
2. Decide whether the convex hulls are tight enough. `Convex Decomposition` is
   the importer's closer fit and is a config change, not a new asset pipeline.
3. Apply `sku2_masses.json` as USD inertia edits when dynamics start to matter.
   The mapping from URDF link names to the mass file's `base` / `<leg>_hip` /
   `<leg>_thigh` / `<leg>_calf` names has to be written down at that point
   (`front_left` is `FL`, `back_left` is `RL`).
4. A stand pose in URDF joint coordinates, derived from `servo_map.json`'s
   `axis_sign` and `zero_pose_deg`, rather than by copying `stand_joint_rad`.
5. Only then an Isaac Lab articulation config, actuators, and a scene.
