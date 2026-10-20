# Vendored BAM (better-actuator-models)

Apache-2.0, see `LICENSE`. Copyright 2025 Marc Duclusaud & Grégoire Passault;
`mangdang/actuator.py` additionally 2026 Mangdang.

Source: the `bam` package of `mangdang-bam/bam` (branch `mangdang`), commit
`13cea06dabd3d537cf160fe42339b349573674f4`.

Vendored: the model and MD01 actuator (`model.py`, `actuator.py`,
`parameter.py`, `testbench.py`, `mangdang/`), the plain MuJoCo binding
(`mujoco.py`, `simulate.py`, `testbench_mujoco.py`), and the mjlab / MuJoCo
Warp binding (`mjlab.py`).

Changes from upstream:

- `bam.*` imports are package-relative; docstrings and messages point at
  `mp4_sim.actuators.bam`.
- `actuators.py` registers only `md01v` and `md01vp`.
- `params/` is not vendored. Fitted parameters live in
  `mp4_sim/actuators/params/` and are passed by path.

Not vendored: the other servo families, servo fitting and bench tools
(`fit.py`, `mae.py`, `plot.py`, `mangdang/tools`, recorders), and their
hardware dependencies.

Known difference: `mjlab.py`'s `_compute_friction_budget` adds the m6
quadratic load-friction term without the opposite-sign condition that
`Model.compute_frictions` in `model.py` applies. The GPU servo
(`mp4_sim.actuators.bam_servo`) inherits this, and the CPU servo bank
(`mp4_sim.actuators.bank.friction_budget`) reproduces it so both backends
train against the same friction.
