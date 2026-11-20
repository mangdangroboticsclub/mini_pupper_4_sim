# The robots and their measured data

`mp4_sim` models two Mini Pupper v4 variants on the same legs, servos and ESP32 board:

| | SKU2 | SKU1 |
|---|---|---|
| Body | long, Raspberry Pi 5 inside, 3S LiPo inside | short |
| Thigh axes, front to rear | 136.6 mm | 97.1 mm |
| Weighed mass | 630.5 g (battery and hood in) | 413.4 g (battery, foot caps and display in) |
| Lidar on the hood (optional) | 43.2 g, 43 mm forward of the base origin | 47.2 g with its mount, 5 mm behind the base origin |
| Package name | `"sku2"` (the default everywhere) | `"sku1"` |

Every function that builds or loads something takes the robot (`robots.get("sku1")`, or the name; left out, the SKU2);
nothing in the package keeps a current robot. A robot is `"sku2"` or `"sku1"` in the code and `sku_2` or `sku_1` as a
folder. The files of each robot are in `src/mp4_sim/robot/assets/sku_<n>/`
([assets README](../src/mp4_sim/robot/assets/README.md)). The model files there are generated from the CAD export by a
tool that is not part of this repository; they are the reference data the policies record. `mp4_sim.robots` records
each file's sha256:

```python
from mp4_sim import robots
robots.check_files("sku1")        # raises ValueError if a model or measured file differs from its recorded hash
```

## Frames and joints

- **Base frame:** x forward, y left, z up; origin at the body's centre (`base` body, the free joint `root`). The IMU is
  site `imu` at the base origin, with the sensors `imu_ang_vel` (gyro), `imu_lin_vel` and `imu_lin_acc` added by
  `mp4_sim.robot.model.get_spec`.
- **Legs:** FR, FL, RR, RL (front right, front left, rear right, rear left).
- **Joints per leg:** `<leg>_hip` (abduction), `<leg>_thigh` (hip, the thigh's swing), `<leg>_calf` (the knee, the shin
  relative to the thigh). The knee servo sits on the hip axis and drives the shin through a rod, modelled as the tendon
  `<leg>_knee_drive`.
- **Joint order** (`joint_order` in the calibration file): FR_hip, FR_thigh, FR_calf, FL_hip, ... RL_calf.
- **Servo order** (ids 1-12, `servo_order`): FR abduction, hip, knee, FL ..., RL. A policy's 12 actions are in servo
  order.
- **Feet:** a sphere of 5.5 mm by default (`Legs(foot="sphere", foot_radius_m=...)`), placed so that at the stand its
  bottom is the mesh foot's lowest point; `foot="mesh"` keeps the model's foot mesh. Site `<leg>_foot` is on the shin.

## The calibration file (`<robot>_contract.json`)

The servo calibration of one robot, as its firmware holds it. `mp4_sim.calibration.Contract.load(robot=...)` reads it
(numpy only; a robot's runtime can copy `calibration.py` alone). The file's `about` field calls it the servo-to-policy
contract; it is the calibration file.

| Key | Meaning |
|---|---|
| `signs` | per servo, the sign from the servo's angle to the model's joint (or tendon) coordinate |
| `stand_joint_rad`, `stand_policy_rad` | the stand pose: all twelve model joints, and the twelve policy targets (joint angles and knee tendon lengths) |
| `offsets_deg` | the calibration offsets stored on the ESP32: the servo angle at which each joint is at the stand |
| `stand_dd`, `stand_ticks` | the stand in the firmware's units (`dd`, deci-degrees on the potentiometer's track; `tick`, the servo command) |
| `tendon_coefs` | how each knee tendon combines the thigh and knee joints |
| `geometry` | the stand's leg geometry: each foot 70 mm below its hip axis (the firmware's neutral pose) |

A servo angle relative to the stand is `rel = sign * radians(servo_deg)`, where `servo_deg` is the angle the firmware's
`servo_write` takes without the offset. The conversions to ticks and `dd` are in `calibration.py`.

The two shipped calibrations are of one SKU2 and one SKU1, the units the measured data below were taken on.

## How the measured data were measured

All on one reference unit of each robot, by hand.

**Masses** (`sku2_masses.json`, `sku1_masses_413g.json`; 413 g is the SKU1's weighed mass). The whole robot on kitchen scales: total; front and rear pair
of feet; left and right pair; and with one end raised on a block (43.0 mm on the SKU2, 43.5 mm on the SKU1). These give
the whole robot's centre of mass (SKU2: x -3.77 mm, y -0.28 mm, z -8.10 mm in the base frame at the stand; SKU1: x
-4.30 mm, y +1.74 mm, z -6.92 mm, 65.93 mm above the floor). A servo module (66.82 g) and one leg's thigh and calf
(6.5 g) were weighed apart. `mp4_sim.robot.model.apply_masses` scales each module and leg to those, gives the rest to the
base, and sets the base's centre of mass so that the model at the stand has the measured whole-robot one. The
CAD masses in `pupper.xml` are only a budget; the file is never edited.

**Joint limits** (`<robot>_joint_limits.json`, its recorded sha256 in `<robot>_joint_limits.sha256`). With the servos
limp, each leg was moved by hand through its range while the firmware streamed the potentiometers at 200 Hz. Per leg:

- the abduction range and the knee's range relative to the thigh, as the robust extremes of the traces (reached in two
  adjacent 2° slices with at least 50 samples each; no margin);
- the hip's potentiometer track ends (the track is 310° with a dead zone beyond: positions there can't be read);
- `<robot>_hip_envelope.json`: where the chassis stops the thigh swinging towards the body's middle, as a function of
  the abduction (one row per 2° of abduction; open where the thigh's run reached its track end, else the traced stop
  less 3°).

On the SKU1 each leg's traces were completed with its left-right partner's, mirrored, inside the leg's own traced
interval; the knee fold is capped at +49° on every leg and each abduction range at the stops marked by hand on that leg;
the hip envelope is capped at every thigh stop marked by hand. Each file's `conventions`, `measured` and per-leg `source`
fields say what each number is. All angles are degrees from the stand: outward abduction, knee folding and hip forward
are positive.

A limits file records the calibration it was traced at (`contract_sha256_prefix`: the first 12 hex digits of the
calibration file's sha256), because the hip's track ends depend on how each joint was mounted and calibrated.
`mp4_sim.limits.limits_contract_check` compares the two; `mp4_sim.limits.load` refuses a file whose sha256 differs from
the recorded one.

**IMU mounting** (`imu_mount.json`): the sensor-to-body rotation (`swap_xy`), found by holding the robot nose down and on
its side and reading gravity. The SKU1 uses the SKU2's.

**Sensor noise** (`noise_measured.json`): per observation element, a uniform half-width from the IMU and the joints at
rest, plus the excess of the robot's readings over the simulation's while trotting in place; the file's `method` field
has the details. The SKU1 uses the SKU2's.

**Servo models** (`actuators/params/`): the MD-300-150 servo module (one module per leg: abduction, hip and knee) fitted
with [BAM](https://github.com/Rhoban/bam): `md300/` per output (the default; BAM's `MD300Actuator`), `md01/` an earlier
fit (BAM's `md01vp` class: MD01 is the module's name in its firmware; `t5ub06` names that parameter set). The module's
AT32 microcontroller runs each output's position loop and reports its potentiometer angle in deci-degrees (`dd`).

## Recorded hashes

Every hash is a sha256 by `mp4_sim.paths.sha256`: text files (`.xml`, `.json`, `.py`, `.csv`, `.md`, `.txt`) with LF line
endings, any other file byte for byte (`.gitattributes` keeps those bytes on checkout).

- `src/mp4_sim/robots.py`: each robot's model, servo map, `SHA256SUMS` and measured files; `robots.check_files` compares
  them.
- `sku_<n>/SHA256SUMS`: the model files and meshes (`sha256sum -c SHA256SUMS`).
- `<robot>_joint_limits.sha256`: the limits file's hash, which `mp4_sim.limits.load` checks on every load.
- `contract_sha256_prefix` in a limits file: the calibration the limits were traced at (above).
- A policy's metadata (in the RL repository) records the model's, the calibration's and the limits' hashes it was
  trained with.

## Adapting to another unit

The geometry and the masses carry over between units of the same robot within the weighing's accuracy. The calibration
does not: every unit has its own servo offsets.

1. **Calibration.** Calibrate the robot with its firmware, read the stored offsets, and write a calibration file with
   the same keys as the shipped one (the RL repository's `tools/make_contract.py` does this from the offsets). Pass it
   where a function takes `contract` or `path`, e.g. `Contract.load(path)`.
2. **Joint limits.** The abduction and knee ranges come mostly from the robot's geometry; the hip's track ends move
   with the calibration. Trace your unit's limbs as above if its ranges are narrower, write them in the same format with
   your calibration's `contract_sha256_prefix`, and record the file's sha256 in the `.sha256` file beside it.
3. **Masses.** Weigh the robot as above and write a masses file with the same keys; pass its path as `masses_json`.

To replace a shipped measured file (for example a later fit of the SKU1's limits), work in a source checkout of this
repository installed in editable mode (`pip install -e .`), not in an installed copy: copy the new file over the old
one under the same name, write its sha256 into `<robot>_joint_limits.sha256` for a limits file, and update the hash in
`src/mp4_sim/robots.py`; `tests/test_assets.py` and `robots.check_files` check the result. The RL repository's
`tools/install_limits.py` checks a limits file and does the copying and the `.sha256`.
