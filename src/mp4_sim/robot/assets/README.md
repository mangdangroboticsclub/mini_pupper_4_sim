# Robot assets

The Mini Pupper v4 models: SKU2 (`sku_2`, the long body) and SKU1 (`sku_1`, the short body on the same legs). Each
`pupper.xml` is a generated MJCF (12 hinges, 4 knee tendons, 12 position actuators, IMU site, `stand` keyframe, ghost
copy `g_*`). `mp4_sim.robots` lists each robot's files with their sha256; `tests/test_assets.py` checks them.

`pupper.xml`, `scene.xml`, `servo_map.json` and `sku_1/parts/base_screen_face.obj` are generated from the CAD export by
a tool that is not part of this repository (their first lines name it and its files). They are the reference data the
policies trained on them record by sha256, and ship exactly as generated.

## Per SKU

| File | What it is |
|---|---|
| `pupper.xml` | The model: 59.75 mm shins, mesh feet (19.3 x 19.6 x 6.0 mm) |
| `scene.xml` | `pupper.xml` plus floor and lights, for the MuJoCo viewer (the SKU1's; the SKU2's needs meshes not shipped, below) |
| `servo_map.json` | Firmware servo angles to model joint angles (`mp4_sim.robot.mapping`) |
| `parts/` | The meshes `pupper.xml` references |
| `urdf/`, `meshes/` | The URDF the model was generated from, and its meshes |
| `measured/` | What was measured on the robot (below) |

SKU2's `parts/` omits the visual-only accessories `pupper.xml` names (`acc_*`, `base_screen_face.obj`,
`screen_face.png`: lidar, Raspberry Pi, display), so the file compiles only through `mp4_sim.robot.model.get_spec`,
which removes them. SKU1's `parts/` holds all 40 meshes and the texture; its front screen (`base_display`,
`base_screen`, `base_screen_face`) and covers are visual only (contype = conaffinity = 0, density 0).
`sku_1/SHA256SUMS` lists every SKU1 model file (`sha256sum -c SHA256SUMS`).

`sku_1/pupper.xml` has the SKU2's legs on the short body: thigh axes 97.1 mm front to rear (136.6 on the SKU2), one base
collision box 144 x 80 x 27 mm. The legs, feet, joint axes, knee tendons and servo-map signs are the SKU2's; the CAD
export's one asymmetric leg is RL (its thigh axis 1.6 mm further out, thigh 50.03 mm).

## Measured data (`measured/`)

| File | What it is |
|---|---|
| `sku2_contract.json`, `sku1_contract.json` | The calibration: servo signs, the stand (all twelve joints and the servo coordinates), calibration offsets, firmware ticks, knee tendon coefficients (`mp4_sim.calibration`) |
| `sku2_masses.json` | The weighed SKU2 (630.5 g): per motor module, per leg, the base takes the rest, the base's centre of mass |
| `sku1_masses_413g.json` | The weighed SKU1 (413.4 g; the name carries the mass), its centre of mass from scales with the rear raised |
| `sku2_joint_limits.json` + `.sha256`, `sku1_joint_limits.json` + `.sha256` | Each robot's joint limits traced on the robot, and the file's recorded sha256 (`mp4_sim.limits`) |
| `sku2_hip_envelope.json`, `sku1_hip_envelope.json` | The hip's stop towards the body's middle as a function of the abduction |
| `imu_mount.json` | The IMU's mounting (sensor to body frame) |
| `noise_measured.json` | Sensor noise measured on the robot, per channel and element |

The SKU1 uses the SKU2's IMU mount and noise profile.

## How a model is loaded

`pupper.xml` carries a 360 g CAD mass budget. `mp4_sim.robot.model.get_spec` reads it and, without editing the file,
removes the ghost, keyframe, XML actuators and sensor and the SKU2 accessories' geoms; adds the IMU (`imu_ang_vel`,
`imu_lin_vel`, `imu_lin_acc` on site `imu`); makes the feet a sphere (`foot_radius_m`, default 5.5 mm, its bottom at the
mesh foot's lowest point at the stand) or keeps the mesh; and applies the robot's masses. The stand comes from the
robot's calibration, not the keyframe. A policy's metadata records the model file's sha256 (`model.legs_meta`).

`servo_profiles.json` (shared) holds a stock position-servo profile (`mp4_sim.actuators.arms.stock`).
