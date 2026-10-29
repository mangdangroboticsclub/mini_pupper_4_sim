# Design: SKU2 as an Isaac asset

`mp4-isaac` imports the Mini Pupper v4 (SKU2) into the current NVIDIA
simulation stack, starting from the CAD URDF. The asset that comes out is a
USD articulation with the CAD meshes, the CAD inertias, the screen, and twelve
leg joints. It is the robot you can open, spawn, and build on inside Isaac Lab.

`mp4-sim` stays the source of the files. This folder does not copy meshes, and
it does not regenerate the MuJoCo model.

## What went wrong last time

`mp4-sim-isaac` started from the compiled MuJoCo model and rebuilt a PhysX
twin of it: a USD authored from MJCF bodies, a knee rotor plus a fixed tendon
for each leg, BAM servo models, and the policy observation contract. That
package was written for Isaac Sim 5.1 / Isaac Lab 2.3 and never simulated
anything. The Windows build on that machine could not start Kit.

Reconstructing MuJoCo features that PhysX does not have (a tendon that carries
the servo armature) is the wrong first step when the goal is a faithful robot
in the NVIDIA stack. The CAD URDF is already the description NVIDIA's importer
is built to read.

## Stack

Isaac Lab 3.0, with the `importers` extra. That extra installs the standalone
URDF importer (`urdf-usd-converter` via `isaacsim-asset-isolated`). Conversion
does not launch Omniverse Kit.

This is the current NVIDIA path (October 2026):

- Isaac Lab 3.0 defaults to Newton. Isaac Sim is an optional extra, and the
  Isaac Sim that extra installs is 6.2. Isaac Sim 5.1 is unsupported on this
  release.
- The importer writes a layered USD: an interface file, geometry payloads, and
  a physics payload that can carry PhysX and MuJoCo attributes beside the
  portable physics description.
- `package://` mesh paths resolve through `ros_package_paths`. The SKU2 URDF
  already uses that form.

A full Isaac Sim / Kit install is out of scope for this pass. NVIDIA does not
support Isaac Sim's RTX renderer under WSL2, and this machine cannot host it:

| | This machine | Isaac Sim workstation ask |
|---|---|---|
| OS | Ubuntu 24.04.2 on WSL2 (host Windows 10 build 19045) | Native Ubuntu 22.04+ or Windows 11 |
| RAM | 7.6 GB in the WSL VM | 32 GB |
| GPU | GeForce RTX 2060, 6 GB | 16 GB, supported RTX |
| Driver seen by WSL | 576.83 (CUDA 12.9) | 580.65.06 or newer for the Lab 3 CUDA 13 wheels |

The C: drive has about 16 GB free, so a Kit install does not fit there either.
The Python environment lives on the Linux filesystem (`/home/afreez/.venvs/mp4-isaac`),
which has the space.

Newton and PyTorch need CUDA for simulation and training. Warp 1.17 ships its
own CUDA 12.9 toolkit and already sees the RTX 2060 on this driver. The Lab 3
wheel's default PyTorch is `2.11.0+cu130`, which refuses this driver, so the
install replaces it with `2.11.0+cu128` (and the matching torchvision). That is
the same PyTorch release, built for a CUDA the driver can load. Isaac Sim's RTX
renderer is a separate requirement and stays out of scope on this machine.

## Source assets

All of these already live under `mp4-sim/src/mp4_sim/robot/assets/sku_2/`.

| Input | Role |
|---|---|
| `urdf/sku_2_urdf_fixed.urdf` | The repaired SolidWorks export. 13 links, 12 continuous joints, metres, x forward, y left, z up. This is the URDF `tools/build_model.py` reads. |
| `meshes/*.STL` | One visual mesh and one collision mesh per link, referenced as `package://sku_2_urdf_fixed/meshes/<link>.STL`. |
| `display/display.json` plus `display_bezel.stl` and `display_panel.stl` | The screen. Visual only. Meshes are already in the base frame. `mp4_sim.robot.display` attaches them to MuJoCo the same way; they are not in the URDF. |
| `servo_map.json`, `sku2_contract.json`, `sku2_masses.json` | Left unread by the importer. They describe the MuJoCo joint convention, the firmware stand, and the weighed 450 g budget. |

The package directory the importer needs is the `sku_2` folder itself.
`package://sku_2_urdf_fixed/meshes/base_link.STL` then resolves to
`sku_2/meshes/base_link.STL`.

SKU1 and the raw CAD export
(`cad/sku_2/sku_2Full_assem_simplified_urdf/`) stay out. The raw export does
not share SKU1's link names or body frame; `tools/fix_sku2_urdf.py` is what
produces the URDF this importer consumes.

## Staging

The importer reads one URDF. The screen is a second asset, so conversion runs
on a generated URDF, written to `mp4-isaac/.cache/urdf/sku_2_with_display.urdf`:

- The repaired URDF, unchanged in kinematics, masses, and mesh paths.
- The MuJoCo coloured pieces instead of one material on each link mesh.
  Those pieces and their colours are `parts/` and the materials in
  `pupper.xml`: yellow body, black frame, dark detail, black motor, metal
  flange, yellow bracket, yellow and dark thigh, dark calf. Collision stays
  the original link STL.
- Two extra links, `display_bezel` and `display_panel`, each with a visual
  mesh, a zero inertial, and no collision. A fixed joint at the identity
  attaches each one to `base_link`. The meshes are already expressed in the
  base frame, which is how `display.add_display` uses them.

`merge_fixed_joints` is on, so those two links fold into `base_link` before
USD conversion. The articulation stays the CAD's 13 links and 12 joints, and
the screen moves with the base.

## Conversion settings

| Setting | Value | Why |
|---|---|---|
| `fix_base` | false | The trunk is a floating base. |
| `merge_fixed_joints` | true | Folds the screen into the base. The leg URDF has no other fixed joints. |
| `collision_type` | Convex Hull | PhysX and Newton need convex collision. The URDF's collision geometry is the full visual STL. |
| `collision_from_visuals` | false | Every link already has a collision mesh. |
| `self_collision` | false | Neighbouring CAD meshes overlap at the joints. Self-collision on those hulls would fight the pose. |
| `link_density` | 0 | Keep the URDF inertias. They are the SolidWorks plastic masses, not the weighed robot. |
| `robot_type` | Quadruped | USD robot schema. |
| `joint_drive.target_type` | none | Gains stay 0. An actuator model can own the joints later. The URDF has no drive of its own. |
| `physics_variant` | physics | The portable physics payload (joints, articulation root, mass). PhysX and MuJoCo payloads are still written; spawn time can select them. |
| `run_asset_transformer` | true | Layered USD, which is the Isaac Lab 3 asset layout. |
| `run_multi_physics_conversion` | true | Write MuJoCo-compatible joint attributes next to the PhysX ones. |
| `make_instanceable` | true | One geometry payload, many spawned copies. |

Continuous URDF joints become revolute joints in USD. The URDF gives them no
limits, and this pass does not invent any.

## What "a good representation" includes

In:

- The thirteen CAD links and twelve CAD joints, with the SolidWorks frames.
- The MuJoCo part meshes, with that yellow, black, metal, and dark split.
- The screen, merged onto the base, with no collision.
- A convex hull per link for collision, taken from the URDF collision mesh.
- The CAD mass and inertia on each link.
- A USD file Isaac Lab can spawn as an articulation, with the portable physics
  variant selected.

Out, on purpose:

- The weighed 450 g / 478 g mass budget (`sku2_masses.json`). The URDF still
  carries the exporter's plastic masses. Replacing them is a later edit to the
  USD inertias, once the visual asset is known to be right.
- The MuJoCo stand pose. `sku2_contract.json` stores angles in the MJCF joint
  convention. `build_model.py` rewrites joint axes (`axis_sign` in
  `servo_map.json`) and drives each knee through a tendon
  (`q_knee_servo = -axis_thigh q_thigh - axis_knee q_calf`). Those angles are
  not URDF joint positions. The USD zero pose is the SolidWorks assembly,
  which `servo_map.json` records as a thigh at about −45° and a shin at about
  45°.
- Knee tendons, rotor joints, BAM servos, observation noise, and the policy
  contract. Those belonged to the MuJoCo-to-PhysX port.
- SKU1.

## Layout

```
mp4-isaac/
  README.md
  docs/DESIGN.md
  docs/IMPLEMENTATION.md
  pyproject.toml
  scripts/install.sh
  src/mp4_isaac/
    paths.py      where mp4-sim's SKU2 files and this cache live
    stage.py      write the URDF the importer reads, and check every mesh
    convert.py    Isaac Lab UrdfConverter -> .cache/usd
    inspect.py    report the staged URDF and, after conversion, the USD prims
  .cache/         generated URDF and USD; not source
```

The environment is not inside this tree. `scripts/install.sh` creates it at
`/home/afreez/.venvs/mp4-isaac`.
