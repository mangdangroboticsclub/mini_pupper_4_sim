# MD01 servo fits

Class `md01vp`, model `m6`.

| File | Servo set | Used for |
|---|---|---|
| `bam_t5ub06_id_3_m6.json` | `id_3` | abduction, knee |
| `bam_t5ub06_id_2_m6.json` | `id_2` | hip |

`t5ub06` is a refit of the published `m6` sets on the same bench data, with
the armature pinned at 0.6 of a bound measured on the bare output arm (knee
2.2e-4, hip 2.7e-4 kg m²) and `kd_ratio` fixed at 1.0. It predicts SKU2's
joints better than the published sets at the robot's light loads.

The role mapping lives in `mp4_sim.actuators.arms.T5UB06`. The three joint
roles are called abduction, hip, and knee for now; the naming is still under
discussion.

Firmware gains (`kp` 80, `kd` 400, `kff` 0.001, `max_pwm` 0.99, 12 V) are
class defaults, not fit parameters, and are not in these files.
