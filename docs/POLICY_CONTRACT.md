# Policy contract

Every policy mp4-sim exports must satisfy this interface. It is the same
contract the robot loads. If the contract changes, this document and the
exporter change with it.

## The ONNX file

| | |
|---|---|
| Inputs | exactly one, float32, shape `[1, 45]` |
| Outputs | exactly one, float32, shape `[1, 12]` |
| Execution provider | `CPUExecutionProvider` |
| Observation normalization | baked into the graph (no separate stats file) |
| Metadata | JSON string under the key `mp3_manifest_json` |

The loader reads tensor names from the graph. The exporter uses `obs` and
`actions`, opset 18, and fixed shapes. The output is the action mean,
unclipped; the runtime clips it.

## Observation (45 values)

| Slice | Content | Scale |
|---|---|---|
| `0:3` | Body angular velocity (IMU gyro), rad/s | × 0.25 |
| `3:6` | Projected gravity, `R_body_to_world.T @ [0, 0, -1]` | × 1 |
| `6:9` | Velocity command `[vx, vy, wz]` (m/s, m/s, rad/s) | × 1 |
| `9:21` | Servo positions, radians relative to the stand pose | × 1 |
| `21:33` | Servo velocities, rad/s | × 0.05 |
| `33:45` | Previous action, after clipping (zeros at reset) | × 1 |

Servo order, for observations and actions:

```
FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
```

`hip` is abduction, `thigh` is the hip servo, and `calf` is the knee servo.

## Action (12 values)

```
action  = clip(output, -1, 1)
targets = A @ (action * 40 degrees) + stand
```

`A` and `stand` come from the SKU's servo map. Policy rate is 50 Hz
(`control_dt = 0.02`); physics runs at 1 kHz.

## Manifest fields a loader requires

A loader rejects the file unless these values in `mp3_manifest_json` match:

| Key | Value |
|---|---|
| `robot` | `MP3-SKU2` (or `MP3-SKU1` for SKU1 events) |
| `task` | `Mjlab-Velocity-Flat-MP3` |
| `observation_dim` | `45` |
| `action_dim` | `12` |
| `control_dt` | `0.02` |
| `action_scale_deg` | `40.0` |

Exports also include `observation_layout`, `observation_scales`,
`servo_order`, `servo_ids`, `action_clip`, `physics_dt`, `normalization`,
and `firmware`, so the file describes itself fully. Loaders may record those
fields without requiring them yet.

Both the GPU and CPU backends export a file that meets this contract.
