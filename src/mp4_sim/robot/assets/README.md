# Robot assets

> **SKU2 URDF to be replaced.** `sku_2/urdf/sku_2_urdf_fixed.urdf` is a
> repaired SolidWorks export. An updated SKU2 URDF will replace it; then put it
> in `sku_2/urdf/`, regenerate with
> `tools/build_model.py --sku sku_2`, and update `sku_2/sku2_masses.json` if
> the new export carries measured masses.

## Per SKU

| File | What it is |
|---|---|
| `pupper.xml` | Generated MJCF: 12 hinges, 4 knee tendons, 12 position actuators, IMU site, `stand` keyframe, ghost copy `g_*` |
| `scene.xml` | `pupper.xml` plus floor and lights, for the viewer |
| `servo_map.json` | Firmware servo angles to model joint angles |
| `urdf/` | Repaired URDF that `tools/build_model.py` reads |
| `meshes/` | CAD meshes referenced by the URDF |
| `parts/` | Split meshes referenced by `pupper.xml` |
| `display/` | The screen (bezel + panel meshes in the base frame, `display.json`): visual only, not in `pupper.xml`; `mp4_sim.robot.display.add_display` attaches it, `sku2.get_spec(strip_visuals=False)` already does |

`servo_profiles.json` (shared) holds the stock position-servo profile.

## SKU2 extras

| File | What it is |
|---|---|
| `sku2_contract.json` | Servo signs, the corrected stand (hip axes 70 mm above the feet), calibration offsets, firmware ticks |
| `sku2_masses.json` | Weighed 450 g split: 66.82 g per motor module, 6.5 g per leg, base takes the rest, base COM x −23.13 mm |
| `sku2_masses_478g.json` | The same with a Raspberry Pi heatsink fitted (478.66 g, the extra on the base); the terrain task's default |
| `imu_mount.json` | Sensor-to-body rotation (`swap_xy`) |
| `noise_n1.json`, `noise_n2.json` | Uniform observation noise profiles: `noise_n1.json` one half-width per channel, `noise_n2.json` one per element (the training default) |

## How SKU2 is loaded

`pupper.xml` still carries SKU1's 360 g mass budget. `mp4_sim.robot.sku2.get_spec`
reads it and, without editing the file, removes the ghost, keyframe, XML
actuators and sensor; adds the IMU (`imu_ang_vel`, `imu_lin_vel`,
`imu_lin_acc` on site `imu`); and applies `sku2_masses.json` (or the mass file
passed as `masses`; the terrain task passes `sku2_masses_478g.json`). The stand
comes from `sku2_contract.json`, not the keyframe.

## Regenerating

```
python tools/fix_sku2_urdf.py            # cad/sku_2/... SolidWorks export -> sku_2/urdf + meshes
python tools/build_model.py --sku sku_2  # urdf + meshes -> pupper.xml, scene.xml, servo_map.json, parts/
```

The screen comes from the SKU 1 SolidWorks assembly and is placed on both SKUs
(SKU 2: centred on its export's placeholder `display_link` box):

```
python tools/export_display.py --assembly path/to/assembly.SLDASM   # Windows + SolidWorks: -> cad/sku_1/display/
python tools/build_display.py                                       # -> sku_1/display/, sku_2/display/
```

It has no mass and no contacts, so regenerating it changes neither
`pupper.xml` nor the contract.

`sku2_contract.json` is derived from the servo map and records the hash of the
`pupper.xml` and `servo_map.json` it was built from. Regenerating either
invalidates it.

## Provenance

Copied from the earlier Mini Pupper simulation project (models, URDFs, meshes,
servo profiles, generator tools) and the walking-policy project (SKU2 contract,
masses, IMU mount, noise profiles). SKU2 `pupper.xml` sha256 (LF):
`7bd8ecc99c261936d04a03d41d17868ad573250d34739d85c809836e467db9b4`.
