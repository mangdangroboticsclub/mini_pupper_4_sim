"""Simulation environments for mjlab (MuJoCo Warp): what a task is built on.

- ``scene``        ``scene_cfg(SceneOptions)``: the robot entity, sensors, terrain and the plant's randomisation events,
                   and the event-ordering contract (``ordered_events``)
- ``terrain``      sub-terrains: low-poly reliefs, the printed tile, measured staircases, rough families
- ``terrain_spawn``  the ground's height under a robot, and spawns on terrain
- ``attachment``   an attachment on the hood (the optional lidar): a collision cylinder and its mass, drawn per robot
- ``com_height``   the whole robot's centre-of-mass height, drawn per robot
- ``contact``      contact-model edits for falls: leg self-collision, stiffer non-foot contacts
- ``noise``        the sensor noise measured on the robot
- ``patch_warp``   a patch to mujoco-warp's heightfield warning (run once after installing)
- ``data/``        measured staircases (``stairs/*.json``) and the printed tile's height map
"""
