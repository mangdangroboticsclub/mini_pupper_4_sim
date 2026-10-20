"""Fitted servo (actuator) models.

- ``bam``        vendored BAM: the MD01 model (NumPy), plus MuJoCo and mjlab bindings
- ``bam_servo``  the GPU servo: BAM on mjlab joints and knee tendons
- ``arms``       actuator sets for training (``stock``, ``bam_t5ub06``)
- ``params/``    fitted parameters (JSON)

Fitting servos on hardware is out of scope; this package only loads and applies the fits.
"""
