# Policy contract

Every policy mp4-sim exports must satisfy this interface. It is the contract
the robot runs: `src/mp4_sim/contract.py` is the reference implementation
(NumPy only, imported by the robot-side program), and the training
environments implement the same arithmetic. If the contract changes, this
document, `contract.py`, and the exporter change together.

The per-SKU numbers (servo signs, stand pose, calibration offsets) live in
`src/mp4_sim/robot/assets/sku_2/sku2_contract.json`.

`python -m mp4_sim.export.verify policies/<name>` checks an exported policy
against this document (see [Checking a policy](#checking-a-policy)).

## The ONNX file

| | |
|---|---|
| Inputs | exactly one, float32, shape `[1, D]`, `D = frame_dim × history` (45 for history 1, 450 for history 10) |
| Outputs | exactly one, float32, shape `[1, 12]` |
| Execution provider | `CPUExecutionProvider` |
| Observation normalization | none, or baked into the graph (no separate stats file) |
| Metadata | JSON string under the key `mp3_manifest_json` |

The loader reads tensor names from the graph. The exporter uses `obs` and
`actions`, opset 18, and fixed shapes. The graph is an MLP: `Gemm` layers
with `Elu` (alpha 1) between them. The output is the action mean,
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

## Observation frame (45 values)

One frame is 45 values (`frame_dim`). No scaling is applied to any slice.

| Slice | Term | Content |
|---|---|---|
| `0:3` | `ang_vel` | Body angular velocity (IMU gyro, bias removed), rad/s |
| `3:6` | `gravity` | Gravity direction in the body frame, `-acc / \|acc\|` from the accelerometer; `(0, 0, -1)` upright |
| `6:9` | `command` | Velocity command. Forward policy: `[vx, 0, 0]`. Omni policy: `[vx, vy, wz]` (m/s, m/s, rad/s), body frame. With a heading loop, `wz` is the loop's output (see [Command source](#command-source)) |
| `9:21` | `pos` | Servo positions, `rel`, rad |
| `21:33` | `vel` | Servo velocities, rad/s, from the 200 Hz positions (`VelocityFilter`: median of 3, then a least-squares slope over 4) |
| `33:45` | `prev_action` | Previous applied action, in action units (zeros at reset) |

## Observation history

A policy with `history` = h observes the last h frames: `D = 45 × h`. With
h = 1 the input is the frame itself.

The layout is term-major (`history_layout: "term_major"`): for each of the
six terms above, in order, that term's h frames from oldest to newest; the
six blocks are concatenated. A term of width w starts at `h × (sum of the
widths before it)`, and frame f (0 oldest, h − 1 newest) of that term is at
`start + f·w : start + (f + 1)·w`. For h = 10:

| Slice | Term | Newest frame |
|---|---|---|
| `0:30` | `ang_vel` | `27:30` |
| `30:60` | `gravity` | `57:60` |
| `60:90` | `command` | `87:90` |
| `90:210` | `pos` | `198:210` |
| `210:330` | `vel` | `318:330` |
| `330:450` | `prev_action` | `438:450` |

After a reset (`history_reset: "repeat_first"`), the first frame fills all h
slots. Each later step drops the oldest frame and appends the new one.
`contract.HistoryStacker` implements this: call `reset()` whenever the
`ActionDecoder` resets, then pass it each 45-value frame
(`assemble_observation`) and feed its output to the policy.

## Action (12 values)

```
a      = clip(output, -1, 1)
rel    = clip(scale * a, -band, +band)                  # about the stand
rel    = clip(rel, prev_rel - 20 deg, prev_rel + 20 deg) # step limit
target = firmware ticks from rel                         # quantised as the robot does
```

- `scale` is per servo, in radians, and is chosen when training; the
  manifest's `action_scale_rad` holds it. The shipped policies use, per leg
  (abduction, hip, knee):

  | Policy | Scale (rad) |
  |---|---|
  | `flat_omni` | `(0.32, 0.45, 0.42)` |
  | `rough_terrain`, `stairs` | `(0.25, 0.50, 0.50)` |
- `band` is 20 degrees for the abductions and 60 degrees for the hips and knees.
- The step limit is the same in simulation and on the robot.
- The previous action observed is the applied target divided by `scale`.

The policy runs at 50 Hz (`control_dt = 0.02`). Physics runs at 1 kHz, and
the robot reports state every 5 ms.

## Command source

The manifest's `command` says where the command slice comes from.

- `{"source": "velocity"}`: the operator's velocity command goes in as is.
- `{"source": "heading_loop", "law": "pi", "kp", "ki", "wz_max", "i_max",
  "antiwindup"}`: the policy was trained to follow a heading. The operator
  sets the forward speed `vx`; `vy` is 0; the robot-side program computes `wz` every
  policy step with `heading.pi_step`, using the manifest's gains:

  ```
  err        = wrap_to_pi(target_heading - heading)     # rad, + = turn left
  u0         = kp * err + ki * integ
  integ_new  = clip(integ + err * dt, -i_max, +i_max)   # dt = control_dt
  if antiwindup == "cond" and |u0| >= wz_max and err * u0 > 0:
      integ_new = integ                                 # hold while saturated
  wz         = clip(kp * err + ki * integ_new, -wz_max, +wz_max)
  integ      = integ_new
  ```

  The target is the heading to hold (for example the heading at start).
  Reset `integ` to 0 when the target changes. The shipped `rough_terrain` and
  `stairs` policies carry `kp 2.0`, `ki 0.5`, `wz_max 0.4` rad/s,
  `i_max 0.5`, `antiwindup "cond"`.

## Manifest fields a loader requires

A loader rejects the file unless these values in `mp3_manifest_json` match:

| Key | Value |
|---|---|
| `robot` | `MP3-SKU2` |
| `observation_dim` | `frame_dim × history` (45 or 450 for the shipped policies) |
| `action_dim` | `12` |
| `control_dt` | `0.02` |
| `action_scale_rad` | the 12 per-servo scales |
| `contract_sha256` | SHA-256 of the shipped `sku2_contract.json` (hashed with LF line endings) |
| `omni` | `true` if the policy observes `[vx, vy, wz]`, `false` for forward-only |

`contract_sha256` names the contract file the policy was trained against.
Calibration offsets are per robot: put a robot's own offsets in the
robot-side program, not in the policy or in the shipped contract file.

These fields are optional; a missing one takes its default:

| Key | Default | Meaning |
|---|---|---|
| `history` | `1` | Frames observed |
| `frame_dim` | `45` | Values per frame |
| `history_layout` | `"term_major"` | The only layout accepted |
| `history_reset` | `"repeat_first"` | The only reset accepted |
| `command` | `{"source": "velocity"}` | `velocity` or `heading_loop` (above) |

Exports may also include `task` (`name`, `actuator`, the actuator model the
policy was trained with, and for an export from a checkpoint `terrain` and
`heading_law`), `observation_layout`, `servo_order`, `action_clip`,
`band_deg`, `step_limit_deg`, `physics_dt`, `normalization` (`"none"`), and,
for a policy exported from a checkpoint, `checkpoint_sha256`, so the file
describes itself fully. Loaders need not check them.

No other keys are allowed, and no value may be an absolute path or a URL.

## Checking a policy

```
python -m mp4_sim.export.verify policies/<name> [policies/<other> ...]
```

On onnxruntime's `CPUExecutionProvider` it checks: one float32 input
`[1, observation_dim]` and one float32 output `[1, 12]`; a valid manifest
whose `contract_sha256` matches the shipped `sku2_contract.json`;
`observation_dim == frame_dim × history`; and that the outputs on the
policy's `fixture.npz` observations are finite and within 1e-5 of the
recorded actions.

The exporter (`python -m mp4_sim.export.onnx`) writes a file that meets
this contract, with its `fixture.npz`.
