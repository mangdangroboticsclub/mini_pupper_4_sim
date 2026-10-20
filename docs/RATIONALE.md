# Why mp4-sim is shaped this way

## The pipeline

```
train (GPU)  ->  checkpoint  ->  ONNX + manifest  ->  robot
```

Anyone with an NVIDIA GPU should be able to run every step on their own
machine and get a policy that loads as-is.

## One project

Simulation, training, and export live in this tree, on one pin set and one
policy contract. A policy trained here is simulated with the same physics it
is scored with, on the observation and action definition the robot uses.

- **Robot and actuators live here.** The SKU1 and SKU2 models (MJCF, meshes,
  servo maps) and the fitted servo models are part of this project. Fitting
  servos on hardware is out of scope; training only needs the fitted models.
- **One definition.** Training runs on the GPU (mjlab, MuJoCo Warp, rsl_rl),
  on the observations and actions specified in
  [POLICY_CONTRACT.md](POLICY_CONTRACT.md). The CPU install provides the
  simulator, servo models and ONNX tooling.
- **One environment.** The GPU and CPU installs share every common version
  (see [INSTALL.md](INSTALL.md)), so physics and export behave the same way
  on every machine.

## Version choices

- **`mujoco 3.7.0`, `mujoco-warp 3.7.0.1`, `warp-lang 1.12.1`, `mjlab 1.3.0`,
  `torch 2.14.0` (CUDA 13.0).** The fitted servo models were identified on
  the MuJoCo 3.7 line, and the shipped policies were trained and evaluated on
  this set, so training and scoring run on the physics the servo models were
  fitted on. Moving a pin means re-evaluating the policies (see
  [INSTALL.md](INSTALL.md#pins-and-why)).
- **Servo fitting stays out.** Hardware tools (Dynamixel, ZMQ, Optuna, PyQt5)
  are separate from this project.

