"""Mini Pupper simulation, locomotion training, and ONNX policy export.

Subpackages:

- ``robot``      robot models (MJCF, meshes, servo maps) per SKU
- ``actuators``  fitted servo models and their MuJoCo / mjlab bindings
- ``cpu``        MuJoCo + Gymnasium environments and Stable-Baselines3 training
- ``gpu``        mjlab tasks and rsl_rl training on MuJoCo Warp
- ``export``     ONNX export, manifest embedding, and verification

``contract`` defines the observation and action interface shared by both
backends (see docs/POLICY_CONTRACT.md); ``heading`` the robot's heading loop.
"""

__version__ = "0.1.0"
