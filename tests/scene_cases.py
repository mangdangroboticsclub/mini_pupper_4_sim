"""Scene options that reproduce the walking, terrain-evaluation and get-up plants, for the scene tests."""
from mp4_sim.envs.attachment import Attachment
from mp4_sim.envs.contact import CONTACT_SOLREF

WALK = dict(masses_json="sku2_masses.json", self_collision=True, foot_solref=(0.01, 1.0), friction_range=(0.5, 1.3),
            base_mass_range=(0.9, 1.1))
LIDAR = Attachment(height_mm=(35.0, 35.0), mass_g=(43.2, 43.2), x_mm=(43.0, 43.0), radius_mm=19.0, present=0.5)
ROUGH_FAMILIES = ("low_poly", "raised_relief", "slope_up", "slope_down", "boxes", "stairs_up", "stairs_down")
FAMILIES = ("flat",) * 9 + ROUGH_FAMILIES + ("stairs4", "stairs8")
CASES = {
    "plane": dict(num_envs=8),
    "rough": dict(num_envs=8, terrain="rough", max_init_level=0, **WALK),
    "flat_limits_mesh": dict(num_envs=8, terrain="flat", rows=2, limits=True, foot="mesh", max_init_level=0, **WALK),
    "stairs": dict(num_envs=8, terrain="stairs_built", stair_layout=((0.025, 0.05),) * 4, stair_width=0.4,
                   max_init_level=0, **WALK),
    "getup_lidar": dict(num_envs=8, terrain="flat", rows=1, max_init_level=0, collisions=True,
                        contact_solref=CONTACT_SOLREF, sensors=("feet_contact",), limits=True, attachment=LIDAR,
                        **WALK),
    "getup_terrain": dict(num_envs=8, terrain="families", families=FAMILIES, rows=8, difficulty_max=0.8,
                          max_init_level=0, collisions=True, contact_solref=CONTACT_SOLREF, sensors=("feet_contact",),
                          **WALK),
    "sku1_getup_com": dict(num_envs=8, robot="sku1", terrain="flat", rows=1, max_init_level=0, collisions=True,
                           contact_solref=CONTACT_SOLREF, sensors=("feet_contact",),
                           attachment=Attachment((35.0, 35.0), (47.2, 47.2), (-5.0, -5.0), 19.0, present=0.5),
                           com_height_mm=(61.0, 70.0), **{**WALK, "masses_json": None}),
}
