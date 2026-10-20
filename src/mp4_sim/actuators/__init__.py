"""Fitted servo (actuator) models.

- ``bam``        vendored BAM: the MD01 model (NumPy), plus MuJoCo and mjlab bindings
- ``fits``       the fitted sets per servo role (t5ub06), their command delays; no mjlab
- ``bam_servo``  the GPU servo: BAM on mjlab joints and knee tendons
- ``bank``       the CPU servo bank: the same servo on plain MuJoCo
- ``arms``       actuator sets for GPU training (``stock``, ``bam_t5ub06``)
- ``params/``    fitted parameters (JSON)

Fitting servos on hardware is out of scope; this package only loads and applies the fits.
"""
