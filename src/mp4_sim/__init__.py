"""Mini Pupper v4 simulation foundation: robot models with measured data, servo models and simulation environments.

Subpackages and modules:

- ``calibration``  the servo calibration (numpy only, no other ``mp4_sim`` import: a robot can copy this file alone)
- ``limits``       measured joint limits: loading, joint ranges, applying them to a model
- ``robots``       the robots modelled (``sku1``, ``sku2``): their files and recorded hashes
- ``robot``        the robot model for MuJoCo / mjlab (``robot.model``), firmware kinematics, servo maps
- ``actuators``    fitted servo models (vendored BAM) for mjlab and plain MuJoCo
- ``envs``         simulation scenes for mjlab: robot, sensors, terrain, plant randomisation, attachments

This module imports nothing.
"""

__version__ = "0.2.0"
