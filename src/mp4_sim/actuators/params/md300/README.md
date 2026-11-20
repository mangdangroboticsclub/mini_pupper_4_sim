# MD-300-150 servo fits

BAM's `MD300Actuator` (Rhoban/bam pull request #26), model `m6`, one parameter set per output of the MD-300-150 servo
module, copied byte for byte from that pull request:

| File | Output | Used for |
|---|---|---|
| `md300_id_1_m6.json` | `id_1` | abduction |
| `md300_id_2_m6.json` | `id_2` | hip |
| `md300_id_3_m6.json` | `id_3` | knee |

These are the default servo model (`mp4_sim.actuators.servo_model`, `md300`). Firmware gains (`kp` 80, `kd` 400,
`pwm_scale` 0.001, `max_pwm` 0.99, 12 V) are set by `choice.MD300_FIRMWARE`, not stored in these files.
