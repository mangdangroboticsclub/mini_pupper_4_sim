"""Fitted servo (actuator) models of MangDang's MD-300-150 servo module (called MD01 in its firmware).

- ``bam``          vendored BAM: the servo models (NumPy), ``MD300Actuator``, and the MuJoCo and mjlab bindings
- ``servo_model``  the actuator set for the mjlab environments: ``md300`` (the default, the robot's servos) or ``md01``
                   (an earlier fit), with the fitted parameters in ``params/md300/`` and ``params/md01/``
- ``choice``       the model's names, gains and flags, without the simulator
- ``bam_servo``    the GPU servo: BAM on mjlab joints and knee tendons, with the per-robot supply model
- ``fits``         the md01 fitted sets per servo role and their command delays; no mjlab
- ``bank``         the CPU servo bank: the same servo on plain MuJoCo
- ``arms``         actuator sets on mjlab: ``stock`` (a position servo) and ``bam_t5ub06`` (the md01 fit)
- ``params/``      fitted parameters (JSON)

Fitting servos on hardware is out of scope; this package only loads and applies the fits.
"""
