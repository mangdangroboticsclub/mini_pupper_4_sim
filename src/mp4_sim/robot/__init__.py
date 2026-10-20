"""Robot models for each SKU.

- ``sku``         SKU registry and where each SKU's files live
- ``sku2``        SKU2 as an mjlab entity: ``pupper.xml`` with the weighed masses and IMU sensors applied
- ``kinematics``  firmware leg kinematics and joint order
- ``mapping``     servo angles <-> model joint angles (``servo_map.json``)
- ``rig``         actuator addressing for the viewer scene (robot + ghost)

Model files live in ``assets/<sku>/``: the MJCF (``pupper.xml``), the viewer
scene, the meshes, the URDF, and the servo map. SKU2 also carries the policy
contract data, the weighed masses, the IMU mount, and the observation noise
profiles. See ``assets/README.md``.
"""
