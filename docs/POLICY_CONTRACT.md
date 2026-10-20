# Policy contract

Every policy mp4-sim exports must satisfy this interface. It is the contract
the robot runs: `src/mp4_sim/contract.py` is the reference implementation
(NumPy only, imported by the robot-side runner), and the training
environments implement the same arithmetic. If the contract changes, this
document, `contract.py`, and the exporter change together.

The per-SKU numbers (servo signs, stand pose, calibration offsets) live in
`src/mp4_sim/robot/assets/sku_2/sku2_contract.json`.

## The ONNX file

| | |
|---|---|
| Inputs | exactly one, float32, shape `[1, 45]` |
| Outputs | exactly one, float32, shape `[1, 12]` |
| Execution provider | `CPUExecutionProvider` |
| Observation normalization | none, or baked into the graph (no separate stats file) |
| Metadata | JSON string under the key `mp3_manifest_json` |

The loader reads tensor names from the graph. The exporter uses `obs` and
`actions`, opset 18, and fixed shapes. The output is the action mean,
unclipped; the runtime clips it.

## Coordinates

Observations and actions are in each servo's own angle, in radians, relative
to the stand pose:

```
rel_i = sign_i * radians(servo_deg_i)
```

`servo_deg` is the firmware's servo angle without the calibration offset
(0 is the stand). The signs and the stand come from `sku2_contract.json`. In
the MuJoCo model the abduction and hip are the `*_hip` and `*_thigh` joints,
and the knee is the `*_knee_drive` tendon (the shin angle).

Servo order, for observations and actions (servo ids 1 to 12):

```
FR abduction, FR hip, FR knee, FL abduction, FL hip, FL knee,
RR abduction, RR hip, RR knee, RL abduction, RL hip, RL knee
```

## Observation (45 values)

No scaling is applied to any slice.

| Slice | Content |
|---|---|
| `0:3` | Body angular velocity (IMU gyro, bias removed), rad/s |
| `3:6` | Gravity direction in the body frame, `-acc / \|acc\|` from the accelerometer; `(0, 0, -1)` upright |
| `6:9` | Velocity command. Forward policy: `[vx, 0, 0]`. Omni policy: `[vx, vy, wz]` (m/s, m/s, rad/s), body frame |
| `9:21` | Servo positions, `rel`, rad |
| `21:33` | Servo velocities, rad/s, from the 200 Hz positions (`VelocityFilter`: median of 3, then a least-squares slope over 4) |
| `33:45` | Previous applied action, in action units (zeros at reset) |

## Action (12 values)

```
a      = clip(output, -1, 1)
rel    = clip(scale * a, -band, +band)                  # about the stand
rel    = clip(rel, prev_rel - 20 deg, prev_rel + 20 deg) # step limit
target = firmware ticks from rel                         # quantised as the robot does
```

- `scale` is per servo, in radians, and is chosen when training. The
  forward walk uses `(0.10, 0.40, 0.42)` per leg (abduction, hip, knee); the
  omni walk uses `(0.32, 0.45, 0.42)`.
- `band` is 20 degrees for the abductions and 60 degrees for the hips and knees.
- The step limit is the same in simulation and on the robot.
- The previous action observed is the applied target divided by `scale`.

The policy runs at 50 Hz (`control_dt = 0.02`). Physics runs at 1 kHz, and
the robot reports state every 5 ms.

## Manifest fields a loader requires

A loader rejects the file unless these values in `mp3_manifest_json` match:

| Key | Value |
|---|---|
| `robot` | `MP3-SKU2` (or `MP3-SKU1` for SKU1) |
| `observation_dim` | `45` |
| `action_dim` | `12` |
| `control_dt` | `0.02` |
| `action_scale_rad` | the 12 per-servo scales |
| `contract_sha256` | hash of the `sku2_contract.json` the policy was trained against |
| `omni` | `true` if the policy observes `[vx, vy, wz]`, `false` for forward-only |

Exports also include `task`, `observation_layout`, `servo_order`,
`action_clip`, `band_deg`, `step_limit_deg`, `physics_dt`, `normalization`,
and `firmware`, so the file describes itself fully. Loaders may record those
fields without requiring them yet.

Both the GPU and CPU backends export a file that meets this contract.
Policies built for the earlier contract (one 40-degree action scale, scaled
gyro and velocity, gravity from the body orientation) are not compatible.
