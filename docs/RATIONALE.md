# Why mp4-sim is shaped this way

## The pipeline

```
train (GPU or CPU)  ->  checkpoint  ->  ONNX + manifest  ->  robot
```

Anyone should be able to run every step on their own machine, with or without
an NVIDIA GPU, and get a policy that loads as-is.

## One project

Simulation, training, and export live in this tree, on one pin set and one
policy contract. A policy trained here is simulated with the same physics it
is scored with, and both backends speak the same observation and action
language.

- **Robot and actuators live here.** The SKU1 and SKU2 models (MJCF, meshes,
  servo maps) and the fitted servo models are part of this project. Fitting
  servos on hardware is out of scope; training only needs the fitted models.
- **Two backends, one definition.** GPU training uses mjlab, MuJoCo Warp, and
  rsl_rl. CPU training uses MuJoCo, Gymnasium, and Stable-Baselines3. Both
  produce observations and consume actions as specified in
  [POLICY_CONTRACT.md](POLICY_CONTRACT.md), and both export through the same
  ONNX path.
- **One environment.** The GPU and CPU installs share every common version
  (see [INSTALL.md](INSTALL.md)), so physics and export behave the same way
  on every machine.

## Version choices

- **`mujoco 3.14.0`, `torch 2.9.1`, `mjlab 1.3.0`, `mujoco-warp 3.8.1`.**
  This set exports ONNX that meets the contract, matches the MuJoCo build
  used when policies are scored, and has been run through train and export
  on Windows.
- **`mujoco-warp` stays on 3.8.1.** That is the version the stack was verified
  with; newer releases are untested. The actuator code passes its unit tests
  on 3.8.1, but run a training job before trusting results (see
  [INSTALL.md](INSTALL.md#known-risks)).
- **Servo fitting stays out.** Hardware tools (Dynamixel, ZMQ, Optuna, PyQt5)
  are separate from this project.

## Still to add

- The robot models and the fitted actuator models.
- The mjlab training stack and the CPU Gymnasium environments, on one shared
  observation and action definition. The CPU observation is 45 values, and
  the action scale is 40° for every joint, as the contract specifies.
- One export command for both backends that embeds the manifest and checks
  PyTorch output against onnxruntime.
