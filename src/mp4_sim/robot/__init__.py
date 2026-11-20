"""Robot models.

- ``model``       a robot (``mp4_sim.robots``) as a MuJoCo spec or an mjlab entity: ``pupper.xml`` with the weighed
                  masses, the calibrated stand, IMU sensors, feet and optional joint limits
- ``sku``         the model folders by SKU name, for the tools below
- ``kinematics``  firmware leg kinematics and joint order
- ``mapping``     servo angles <-> model joint angles (``servo_map.json``)
- ``rig``         actuator addressing for the viewer scene (robot + ghost)

Model files live in ``assets/<sku>/``: the MJCF (``pupper.xml``), the viewer scene, the meshes (``parts/``), the URDF
and its meshes, the servo map, and ``measured/``: the calibration, masses, joint limits, IMU mount and sensor noise
measured on the robot. See ``assets/README.md``.
"""
