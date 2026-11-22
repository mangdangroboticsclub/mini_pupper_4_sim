"""The simulation scene for a Mini Pupper v4 in mjlab: the robot entity, its sensors, the terrain, and the plant's
randomisation events.

``scene_cfg(SceneOptions)`` returns a ``Scene``: an mjlab ``SceneCfg`` (the robot ``"robot"`` with its servo models,
the IMU in the model, the contact and foot-height sensors, a plane or a generated terrain) and the startup events that
make each simulated robot a different plant. A task (observations, actions, commands, rewards, terminations,
curricula) is built on top of it.

**The robot** (``robot.model.robot_cfg``): the options' robot, legs and feet, masses, shin and foot <-> base contacts
(``self_collision``), the SKU1's display collision box (``display_collision``), the feet's contact ``foot_solref``, the
measured joint limits (``limits``), and the servo model (``actuators.servo_model``: ``actuator`` with the optional
supply randomisation). On top, in this order: the get-up's contact edits (``envs.contact``: leg self-collision
``collisions``, ``thigh_base`` unless ``limits``, stiffer non-foot contacts ``contact_solref``), then an attachment on
the hood (``envs.attachment``).

**Sensors** (``sensors``, by name, in the order given; ``SENSORS``): ``feet_contact`` (feet against the terrain:
found, net force, air time), ``body_contact`` (the trunk and the motor cans against the terrain), ``body_scan`` (a grid
of rays under the base), ``foot_height`` (each foot site's height above the terrain under it), ``leg_contact`` (the
thigh and shin capsules against the terrain). The IMU's sensors are in the model (``robot.model.get_spec``).

**Terrain** (``terrain``): ``"plane"``, or a generator of ``rows`` difficulty rows by columns of sub-terrains:
``"rough"``, ``"flat"``, ``"families"`` (``families``), ``"square_walls"`` (the printed tile), ``"stairs_built"`` (one
measured staircase, ``stair_layout``; ``stair_layout_file`` reads one from ``data/stairs/``), or the caller's own
``sub_terrains``.

**Events and their order.** mjlab runs the events of one mode in the order of its event dict, and the random draws
follow that order, so the order is part of the plant. ``Scene.events`` and ``Scene.late_events`` are ordered dicts of
startup events:

- ``events``: ``bam_init`` (the servo models' state, the supply draws), then with ``randomise``: ``foot_friction``
  (the feet's sliding friction, one draw per robot), ``base_com`` (the base's centre of mass offset), ``base_mass``
  (the base's mass scaled; ``base_mass_range`` not None);
- ``late_events``: ``attachment`` (the hood attachment's draw, combined into the base), then ``com_height`` (the whole
  robot's centre-of-mass height, ``com_height_mm``). Both build on the earlier draws.

A task composes its event dict with ``ordered_events``: its own first events (in the base walking environment the
observation-noise install, then the resets ``reset_base``, ``reset_joints``, ``command_delay``), then
``Scene.events``, then the task's other events (startup: e.g. a foot-softness draw; reset: e.g. fallen starts; step
and interval: pushes), then ``Scene.late_events``. Any other order changes the random stream, and with it every
seeded rollout.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensorCfg, RingPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from mp4_sim import paths, robots
from mp4_sim.actuators import servo_model
from mp4_sim.actuators.bam_servo import bam_servo_init
from mp4_sim.envs import attachment as attachment_mod
from mp4_sim.envs import com_height as com_height_mod
from mp4_sim.envs import contact
from mp4_sim.envs import terrain as terrain_mod
from mp4_sim.robot.model import (
    BODY_CONTACT_BODIES,
    FOOT_GEOMS,
    FOOT_RADIUS_M,
    FOOT_SITES,
    LEG_COLLISION_GEOMS,
    Legs,
    foot_site_clearance,
    legs_of,
    robot_cfg,
)

BORDER = 0.5                                    # m: the generated terrain's border
SINGLE_ROW = ("square_walls", "stairs_built")   # terrains built as one row
TERRAINS = ("plane", "rough", "flat", "square_walls", "stairs_built", "families")
SENSORS = ("feet_contact", "body_contact", "body_scan", "foot_height", "leg_contact")
PLANE_SENSORS = ("feet_contact", "body_contact")
TERRAIN_SENSORS = ("feet_contact", "body_contact", "foot_height", "leg_contact")
BASE_COM_RANGES = {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.003, 0.003)}   # m, added to the base's centre of mass
STAIRS_DIR = paths.ENV_DATA / "stairs"


def _fresh_raycasts() -> None:
    """mjlab 1.3.0 caches a sensor's data until ``reset`` or ``update``, but raycasts are refreshed later, in
    ``sense()``, so a reader between the two gets the previous step's rays (upstream issue 998, fixed in 1.5.3).
    Invalidate the cache when ``sense()`` writes new rays."""
    from mjlab.sensor.raycast_sensor import RayCastSensor
    if getattr(RayCastSensor, "_fresh_rays", False):
        return
    original = RayCastSensor.postprocess_rays

    def postprocess_rays(self):
        original(self)
        self._invalidate_cache()

    RayCastSensor.postprocess_rays = postprocess_rays
    RayCastSensor._fresh_rays = True


_fresh_raycasts()


@dataclass
class SceneOptions:
    num_envs: int = 4096
    env_spacing: float = 1.0
    # the robot (robot.model.Legs; robots)
    robot: str = robots.DEFAULT                 # "sku2" or "sku1"
    legs: str = "sku_2"                         # the leg set (59.75 mm shin)
    foot: str = "sphere"                        # "sphere" (foot_radius_m) or "mesh" (the model's)
    foot_radius_m: float = FOOT_RADIUS_M
    masses_json: str | None = None             # the masses file (robot.model.masses_path; None: the robot's)
    self_collision: bool = False                # shins and feet collide with the base
    foot_solref: tuple | None = None            # the feet's contact (timeconst, dampratio); None: MuJoCo's
    limits: bool = False                        # the measured joint limits as joint ranges (limits.apply_limits)
    visuals: bool = False                       # keep the visual meshes (rendering only)
    display_collision: bool = False             # the SKU1's display as a collision box (robot.model.add_display_collision)
    # the servos (actuators.servo_model)
    actuator: str = servo_model.NAME            # "md300" or "md01"
    supply_v_range: tuple | None = None         # per-robot servo supply voltage, uniform (V); None: 12 V
    supply_drop_ohm_range: tuple | None = None  # per-robot pack + wiring resistance (Ohm)
    r_scale_range: tuple | None = None          # per-robot winding resistance scale (heat)
    # contact edits (envs.contact), applied in this order after the robot's own spec
    collisions: bool = False                    # leg <-> leg, leg <-> hip can (and thigh <-> base: thigh_base)
    thigh_base: bool = True                     # with collisions and without limits: thigh <-> base
    thigh_short_m: float = contact.THIGH_SHORT_M
    contact_solref: tuple | None = None         # stiffer contacts on every collision geom but the feet; None: off
    # the terrain
    terrain: str = "plane"                      # TERRAINS
    sub_terrains: dict | None = None            # the generator's columns (overrides terrain's)
    rows: int = 10                              # difficulty levels (1 for SINGLE_ROW terrains)
    families: tuple | None = None               # "families": the columns (terrain.family_columns)
    difficulty_max: float = 1.0                 # the last row's upper difficulty
    relief_pitch: float | None = None           # "families": the reliefs' heightfield pitch (m)
    tile: float = terrain_mod.TILE_SIZE
    terrain_seed: int = 0
    max_init_level: int | None = 2              # the generator's max_init_terrain_level
    stair_layout: tuple | None = None           # "stairs_built": ((riser, tread), ...) in m
    stair_width: float | None = None            # "stairs_built": the treads' width (m; None: the whole tile)
    stair_landing: float | None = None          # "stairs_built": the top tread's length (m; None: to the tile's edge)
    # sensors
    sensors: tuple | None = None                # names from SENSORS; None: PLANE_SENSORS or TERRAIN_SENSORS
    # the plant's randomisation (startup events)
    randomise: bool = True
    friction_range: tuple = (0.4, 1.2)          # the feet's sliding friction
    base_com_ranges: dict = field(default_factory=lambda: dict(BASE_COM_RANGES))
    base_mass_range: tuple | None = None        # the base's mass scale; None: no draw
    attachment: attachment_mod.Attachment | None = None   # a cylinder on the hood and its mass, drawn per robot
    attachment_top_z: float | None = None       # the hood's top (base frame, m); None: the robot's top_z_m
    com_height_mm: tuple | None = None          # the whole robot's CoM height range (envs.com_height); None: off


@dataclass
class Scene:
    scene: SceneCfg
    events: dict                    # the plant's startup events, in order (module docstring)
    late_events: dict               # startup events after every other startup event, in order
    servo: servo_model.ServoModel   # the servo model (its latency_steps: each actuator group's command latency)
    legs: Legs
    sub_terrains: dict | None       # the generator's columns (None: a plane)
    rows: int                       # the generator's rows (1 for a plane)
    half_x: float | None            # the terrain's half extent (m, with the border; None: a plane)
    half_y: float | None
    foot_site_clearance: float      # a foot site's height above flat ground at a foot contact (m)


def stair_layout_file(name: str) -> dict:
    """A measured staircase in ``data/stairs/`` (``four_steps``, ``eight_steps``): its JSON (``steps_mm``: (riser,
    tread) per step, ``width_mm``)."""
    p = STAIRS_DIR / (name if name.endswith(".json") else f"{name}.json")
    return json.loads(p.read_text())


def sub_terrains(opt: SceneOptions) -> dict:
    """The generator's columns for ``opt.terrain`` (or ``opt.sub_terrains``)."""
    if opt.sub_terrains is not None:
        return dict(opt.sub_terrains)
    if opt.terrain == "rough":
        return terrain_mod.rough_sub_terrains(opt.tile)
    if opt.terrain == "families":
        return terrain_mod.family_columns(opt.families, opt.tile, opt.relief_pitch)
    if opt.terrain == "square_walls":  # four blocks of the printed tile, every edge a mirror line, with vertical sides
        return {"square_walls": terrain_mod.TileStripCfg(size=(opt.tile, opt.tile), n_long=2, n_wide=2, walls=True)}
    if opt.terrain == "stairs_built":  # one measured staircase
        return {"built": terrain_mod.StairGridCfg(size=(opt.tile, opt.tile), layout=tuple(opt.stair_layout),
                                                  width=opt.stair_width, landing=opt.stair_landing, tag="built")}
    if opt.terrain == "flat":
        return {"flat": BoxFlatTerrainCfg(size=(opt.tile, opt.tile))}
    raise ValueError(f"unknown terrain {opt.terrain!r}")


def sensor_cfg(name: str):
    """One sensor of the catalogue (``SENSORS``; module docstring)."""
    if name == "feet_contact":
        return ContactSensorCfg(name="feet_contact",
                                primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
                                secondary=ContactMatch(mode="body", pattern="terrain"),
                                fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True)
    if name == "body_contact":
        return ContactSensorCfg(name="body_contact",
                                primary=ContactMatch(mode="body", pattern=BODY_CONTACT_BODIES, entity="robot"),
                                secondary=ContactMatch(mode="body", pattern="terrain"),
                                fields=("found",), reduce="none", num_slots=1)
    if name == "body_scan":
        return RayCastSensorCfg(name="body_scan", frame=ObjRef(type="body", name="base", entity="robot"),
                                ray_alignment="yaw", pattern=GridPatternCfg(size=(0.30, 0.18), resolution=0.03),
                                max_distance=1.0, exclude_parent_body=True, include_geom_groups=(0,))
    if name == "foot_height":
        return TerrainHeightSensorCfg(name="foot_height",
                                      frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
                                      pattern=RingPatternCfg.single_ring(radius=0.006, num_samples=4),
                                      ray_alignment="yaw", max_distance=0.5, exclude_parent_body=True,
                                      include_geom_groups=(0,), reduction="min")
    if name == "leg_contact":                  # thigh and shin capsules, not the feet
        return ContactSensorCfg(name="leg_contact",
                                primary=ContactMatch(mode="geom", pattern=LEG_COLLISION_GEOMS, entity="robot"),
                                secondary=ContactMatch(mode="body", pattern="terrain"),
                                fields=("found",), reduce="none", num_slots=1)
    raise ValueError(f"sensor: one of {SENSORS}, not {name!r}")


def plant_events(opt: SceneOptions) -> dict:
    """The plant's startup events, in order: ``bam_init``, and with ``randomise`` ``foot_friction``, ``base_com``,
    ``base_mass``."""
    events = {"bam_init": EventTermCfg(func=bam_servo_init, mode="startup")}
    if opt.randomise:
        events["foot_friction"] = EventTermCfg(
            func=envs_mdp.dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                    "axes": [0], "ranges": tuple(opt.friction_range), "shared_random": True})
        events["base_com"] = EventTermCfg(
            func=envs_mdp.dr.body_com_offset, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "add",
                    "ranges": dict(opt.base_com_ranges)})
        if opt.base_mass_range is not None:
            events["base_mass"] = EventTermCfg(
                func=envs_mdp.dr.body_mass, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "scale",
                        "ranges": tuple(opt.base_mass_range)})
    return events


def scene_cfg(opt: SceneOptions | None = None) -> Scene:
    """The scene and the plant's events for ``opt`` (module docstring)."""
    opt = opt or SceneOptions()
    if opt.terrain not in TERRAINS and opt.sub_terrains is None:
        raise ValueError(f"terrain: one of {TERRAINS}, not {opt.terrain!r}")
    legs = legs_of(opt)
    r = robots.get(legs.robot)
    servo = servo_model.for_options(opt)

    robot = robot_cfg(servo.actuators, strip_visuals=not opt.visuals, masses_json=opt.masses_json,
                      self_collision=opt.self_collision, foot_solref=opt.foot_solref, limits=opt.limits, legs=legs,
                      display_collision=opt.display_collision)
    if opt.collisions or opt.contact_solref is not None:
        robot.spec_fn = contact.wrap_spec_fn(robot.spec_fn, opt.collisions, opt.thigh_base and not opt.limits,
                                             opt.thigh_short_m,
                                             None if opt.contact_solref is None else tuple(opt.contact_solref))

    late = {}
    if opt.attachment is not None:
        top = r.top_z_m if opt.attachment_top_z is None else float(opt.attachment_top_z)
        robot.spec_fn = attachment_mod.wrap_spec_fn(robot.spec_fn, opt.attachment, top)
        late["attachment"] = attachment_mod.event_cfg(opt.attachment, top)
    ev = com_height_mod.event_cfg(opt)
    if ev is not None:
        late[com_height_mod.EVENT] = ev

    if opt.terrain == "plane" and opt.sub_terrains is None:
        subs, rows, half_x, half_y = None, 1, None, None
        terrain = TerrainEntityCfg(terrain_type="plane")
        default_sensors = PLANE_SENSORS
    else:
        subs = sub_terrains(opt)
        rows = 1 if opt.terrain in SINGLE_ROW else opt.rows
        gen = TerrainGeneratorCfg(size=(opt.tile, opt.tile), sub_terrains=subs, num_rows=rows, curriculum=True,
                                  border_width=BORDER, seed=opt.terrain_seed,
                                  difficulty_range=(0.0, opt.difficulty_max))
        terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=gen,
                                   max_init_terrain_level=opt.max_init_level)
        half_x = rows * opt.tile / 2 + BORDER
        half_y = len(subs) * opt.tile / 2 + BORDER
        default_sensors = TERRAIN_SENSORS
    sensors = tuple(sensor_cfg(n) for n in (default_sensors if opt.sensors is None else opt.sensors))

    scene = SceneCfg(terrain=terrain, num_envs=opt.num_envs, env_spacing=opt.env_spacing, entities={"robot": robot},
                     sensors=sensors)
    return Scene(scene=scene, events=plant_events(opt), late_events=late, servo=servo, legs=legs, sub_terrains=subs,
                 rows=rows, half_x=half_x, half_y=half_y, foot_site_clearance=foot_site_clearance(legs))


def ordered_events(scene: Scene, first: dict | None = None, middle: dict | None = None) -> dict:
    """A task's event dict in the order the plant's random stream depends on: ``first`` (the task's first events),
    ``scene.events``, ``middle`` (the task's other events), ``scene.late_events``. A name in two parts is an error."""
    out: dict = {}
    for part in (first or {}, scene.events, middle or {}, scene.late_events):
        for k, v in part.items():
            if k in out:
                raise ValueError(f"event {k!r} given twice")
            out[k] = v
    return out


def foot_heights_centre(env, sensor_name: str = "foot_height"):
    """Each foot site above the terrain straight under it (the ring's centre ray; the site's current height, the ray from
    the previous ``sense()``); 0 when the ray missed. A foot touches flat ground with its site at
    ``Scene.foot_site_clearance``."""
    import torch
    d = env.scene[sensor_name].data
    B, F = d.heights.shape[0], d.heights.shape[1]
    hit_z = d.hit_pos_w[..., 2].view(B, F, -1)[..., 0]
    hit = d.distances.view(B, F, -1)[..., 0] >= 0
    site = env.scene["robot"].data.site_pos_w[:, foot_site_ids(env), 2]
    return torch.where(hit, site - hit_z, torch.zeros_like(site))


def foot_site_ids(env):
    """The robot's foot sites' indices (``FOOT_SITES`` order), cached on the env."""
    robot = env.scene["robot"]
    if not hasattr(env, "_foot_sites"):
        env._foot_sites = [robot.site_names.index(s) for s in FOOT_SITES]
    return env._foot_sites
