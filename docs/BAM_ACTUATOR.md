# BAM actuators and weighed masses

The Mini Pupper in Isaac should weigh what MuJoCo weighs, and each servo
should be the same BAM model the policies were trained on. Those are separate
edits. The mass is a property of the rigid bodies. The servo is an explicit
Isaac Lab actuator. Isaac's own joint friction is not that servo.

[BAM](https://github.com/Rhoban/bam) identifies friction as a budget, not as a
single Coulomb coefficient. The variants are cumulative:

| Model | What it adds |
|---|---|
| m1 | Coulomb plus viscous |
| m2 | Stribeck, extra friction near zero speed |
| m3 | Load-dependent friction |
| m4 | Stribeck crossed with load |
| m5 | Motor-side load versus external-side load |
| m6 | A quadratic load term on top of m5 |

Isaac Lab's implicit actuator hands a constant friction and a viscous damping
to the solver. That is m1. The SKU2 policies were trained on m6
(`bam_t5ub06` in `mp4_sim.actuators.fits`). m6's budget depends on the motor
torque and on the external load at that step, so it cannot be a pair of
numbers written once into the joint.

## Masses

Do this first, and do it without touching the actuator. The USD still carries
the SolidWorks plastic. A stiff servo on that inertia is what flipped the
robot.

SKU2's budget is `sku2_masses.json` (450 g) or `sku2_masses_478g.json` (the
heatsink, 478.66 g, the terrain task's default). `mp4_sim.robot.sku2.apply_masses`
already defines the split. Repeat it on the Isaac bodies after spawn:

| MuJoCo body | Isaac link | Mass |
|---|---|---|
| `base` | `base_link` | the remainder, after the modules and the legs |
| `<leg>_hip` | `<prefix>_hip_link` | `module_kg` (66.82 g) |
| `<leg>_thigh` | `<prefix>_leg_upper_link` | the leg's 6.5 g split by the CAD thigh/calf ratio |
| `<leg>_calf` | `<prefix>_leg_lower_link` | the rest of that 6.5 g |

`<prefix>` is `front_right`, `front_left`, `back_right`, `back_left` for
`FR`, `FL`, `RR`, `RL`. Scale each link's inertia by the new mass over the
CAD mass, about the existing centre of mass. Then set the base centre of
mass x to `base_com_x_m` (−23.13 mm). The screen is merged into the base and
has no mass of its own.

SKU1 has no separate mass file. Its MuJoCo model was generated at 360 g with
65.5 g on each hip module (`tools/build_model.py`). Read those body masses
and inertias out of `sku_1/pupper.xml` and write the same numbers onto the
matching links. Do not apply the SKU2 budget to SKU1.

Leave the URDF alone. The budget is chosen at spawn, the way MuJoCo chooses
`sku2_masses.json` versus `sku2_masses_478g.json` without editing `pupper.xml`.

## The actuator Isaac should run

`isaac_bam_actuators` is already the right shape. `BamActuator` is an
explicit `ActuatorBase`: `is_implicit_model` stays false, `compute` fills
`joint_efforts`, and it clears `joint_positions` and `joint_velocities` so
the solver does not also run a PD. `BamFrictionModel` is the torch port of
BAM's `compute_frictions`, including m6. The motor object is the firmware
law.

Use that class. Do not reimplement m6 inside `mp4-isaac`, and do not turn
stiffness and damping on an `ImplicitActuatorCfg` up until the walk looks
right. Those two numbers are m1.

The fits to load are the ones the policies used, not the pendulum files
shipped beside `md01i`:

| Role | Joints | Params | Law |
|---|---|---|---|
| abduction | the four `*_hip_joint` | `assets/actuators/bam_t5ub06_id_3_m6.json` | `md01vp` |
| hip | the four `*_leg_upper_joint` | `bam_t5ub06_id_2_m6.json` | `md01vp` |
| knee | the servo coordinate below | `bam_t5ub06_id_3_m6.json` | `md01vp` |

`md01vp` is AT32 `control_mode` 1: position PID to PWM duty, then a brushed
DC plant with its own back-EMF constant. The bundled `md01i` current-law fits
are a different bench and must not be substituted. Each JSON's `"model": "m6"`
selects the friction flags. Firmware gains (`kp` 80, `kd` 400, `kff` 0.001,
`max_pwm` 0.99, 12 V) are class defaults of that law, not fields in the JSON.

One explicit actuator group owns all twelve hinges, because the knee torque
lands on the thigh as well as the calf and a joint can belong to only one
group. Joint friction on the USD stays at zero. The JSON armature
(2.7×10⁻⁴ kg m² on the hip fit, 2.2×10⁻⁴ on the abduction and knee fits) is
the rotor inertia MuJoCo puts on the hinge or the knee tendon, so the solver
gets that diagonal: abduction on the hip joint, the hip rotor plus the knee
tendon's share on the thigh, the rest of the tendon on the calf. The stopping
clip keeps the same servo armature and uses it only to size stiction. Leaving
it out of the solver integrates a servo-sized torque against the CAD link
inertia and the joints run away.

## What one step computes

Per environment and per servo, matching `BamActuator.compute` and BAM's own
step:

1. The firmware law turns the position target into a duty or a current.
2. The motor law turns that into `motor_torque` (back-EMF included).
3. m6 turns `(motor_torque, external_load, velocity)` into a friction budget:
   a Coulomb/Stribeck/load piece plus a viscous coefficient.
4. The stopping torque is `motor_torque + (armature / dt) * velocity`.
   Friction opposes it, clipped to the budget.
5. The effort written to the solver is `motor_torque + friction`, then
   clipped to the servo's torque limit.

The solver already applies gravity, Coriolis, and contact. The external load
is an input to the friction model only. It must not be added into
`joint_efforts`. The current `computed_effort` line adds it. On a pendulum
whose `set_external_torque` is zero that is harmless. On this robot it
applies gravity a second time. Split the two: log `motor_torque`, and write
only `motor_torque + friction`.

`physics_dt` on the cfg is the simulation step. The stopping clip divides by
it, and the fitted command delays are a few milliseconds, so the step should
be 1 ms, which is also MuJoCo's `PHYSICS_DT`. A 5 ms step changes both the
friction clip and the delay quantisation.

## The knee is a servo coordinate, not a second hinge

MuJoCo puts the knee servo on a fixed tendon. The tendon coordinate is the
shin angle, `c_thigh * q_thigh + c_calf * q_calf`, and the servo's armature
and friction sit on that coordinate. The calf hinge itself is passive.
`ServoMap` is the same relationship, already used to aim `flat_omni` at the
CAD hinges.

Isaac has no tendon transmission on an actuator. Do not add the rotor joint
the old PhysX port used. Keep one coordinate inside the actuator:

- Position and velocity that `md01vp` and m6 see are the servo angle and its
  rate, from `ServoMap.joint_to_servo` on the thigh and calf (and the hip
  joint alone for the other two roles).
- The scalar effort comes back through the same linear map: torque on the
  thigh and torque on the calf equal the servo torque times that row of the
  map. Abduction and hip apply their torque to one joint.
- The armature in the stopping clip is the JSON armature, in that servo
  coordinate.

The policy's 12 actions stay in servo order. The actuator's 12 channels are
those servos. The calf joint is not a thirteenth command.

## External load

m3 through m6 need the torque the gearbox sees from the leg. MuJoCo reads it
from `qfrc_bias` and the constraint forces, projected onto the joint or the
tendon. Newton does not publish that vector.

Estimate it after each accepted step, in the servo coordinate:

```
load = armature * (v - v_prev) / dt - motor_torque_prev
```

`v` is the servo velocity. This is the load the rotor dynamics imply. It
lags one step, which is the same lag the previous Isaac port accepted. Check
it on the stand: holding friction and the breakaway when a joint starts to
move should look like the MuJoCo `bam_servo` under the same hold, not like a
joint that is either glued or frictionless.

If that residual is wrong because the link inertia dominates the armature,
the next measurement is the generalized force at the joint from the
solver, minus the effort this actuator wrote. Do not invent a second
friction term on the link while that comparison is open.

## Delays

The T5 ub06 fits carry `command_delay`: about 5 ms on the hip (`id_2`) and
about 1 ms on abduction and knee (`id_3`), rounded up to physics steps.
`BamActuator` currently prints that value and does not apply it. The buffer
belongs in the actuator: a command becomes the firmware target on the step
where its delay has elapsed. The motor law itself stays stateless.

The 8–19 ms sensing-to-command chain delay is not in the fit. MuJoCo puts it
on the action term, once per episode. Leave it there when a task exists.
Playback of `flat_omni` can keep it at zero until the delays are being
compared on purpose.

## What has to be checked before it is trusted

`isaac_bam_actuators` was written against Isaac Lab 2.3's `ActuatorBase`.
Lab 3 still has the same `compute(control_action, joint_pos, joint_vel)`
signature and still defaults `is_implicit_model` to false, but the
constructor's effort-limit fields moved. Port the cfg onto Lab 3's
`ActuatorBaseCfg` (`joint_effort_limit`, not a solver PD). The offline
pytest suite (friction, motors, no Isaac) should stay green; it does not
prove the Lab 3 wiring.

Then, in Newton, one servo at a time, same position command as MuJoCo's
`bam_servo`:

- Position, velocity, and motor torque over a short move, before the leg is
  attached. A single-joint rig with the JSON armature.
- The same servo on the robot at the stand: it holds, and the base stays
  near the MuJoCo stand height once the weighed mass is on.
- `flat_omni` at 0.15 m/s. The current plain PD covers 0.29 m in 3 s and
  only does that because the gain is far below the servo. This actuator
  should be allowed to be compared with MuJoCo's own evaluation of that
  policy, not with the soft PD.

SKU1 uses the same three role fits only if its servos are the same MD01s.
The mass write for SKU1 is the 360 g model, independent of that choice.

## Out of this design

Observation noise, the heading loop, terrain, and PPO. They sit on top of a
robot whose weight and servos already match. Collision primitives (the
MuJoCo boxes, cans, capsules, and foot spheres) are still not these masses
and not this actuator.
