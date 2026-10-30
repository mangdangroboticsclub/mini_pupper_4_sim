"""SKU2 velocity tracking in Isaac Lab on Newton (MuJoCo Warp), for PPO.

The robot is the converted USD (``convert.py``) with the weighed masses (``masses.py``) and the BAM servo model
(``servos.py``) on every hinge. Physics runs at 1 ms and the policy at 50 Hz (decimation 20). The actor
observes ``contract`` (``mdp.py``), optionally with a term-major history, so a checkpoint
exports to the ONNX contract in this package.

Terrains: ``flat`` (a plane) or ``rough`` (Isaac Lab's generator at Mini Pupper scale: relief, boxes, pyramid stairs
and slopes, 10 difficulty rows, promoted by distance walked).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as lab_mdp
from isaaclab.managers import CurriculumTermCfg, EventTermCfg, ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab import terrains as terrain_gen

from . import mdp
from .paths import PHYSICS_DT, interface_usd
from .servos import PupperBamActuatorCfg

DECIMATION = 20
TERRAINS = ("flat", "rough")


@dataclass
class TaskOptions:
    num_envs: int = 1024
    terrain: str = "flat"
    history: int = 0                                    # actor frames; 0 or 1: the 45-value observation
    action_scale: tuple = (0.25, 0.50, 0.50) * 4
    vx_max: float = 0.15
    vy_max: float = 0.0
    wz_max: float = 0.6
    rel_standing: float = 0.1
    resample_s: tuple = (5.0, 10.0)
    heading: str = "none"                               # "loop": most envs follow a heading (P law, clip to wz_max)
    heading_kp: float = 2.0
    rel_heading: float = 0.75
    episode_s: float = 20.0
    spawn_z: float = 0.10                               # base height at reset, m (the stand's feet just above ground)
    heatsink: bool = True                               # 478 g (Raspberry Pi heatsink); False: 450 g
    noisy: bool = True
    noise: str = "n2"
    noise_ramp: bool = True
    noise_hold_iters: int = 150
    noise_ramp_iters: int = 300
    randomise: bool = True
    friction: tuple = (0.5, 1.3)
    push_mps: float = 0.05
    chain_ms: tuple = mdp.CHAIN_MS
    tilt_limit: float = 0.7
    progress: float = 0.0
    abduction: float = 0.0
    rows: int = 10
    max_init_level: int = 2
    solver_iterations: int = 10
    solver_ls_iterations: int = 20
    njmax: int = 200
    nconmax: int = 64
    fixed_command: tuple | None = None                  # (vx, vy, wz) for every env (evaluation)
    extra: dict = field(default_factory=dict)


def usd_path() -> str:
    path = interface_usd("sku_2")
    if path is None:
        raise SystemExit("no USD for sku_2 yet; run python -m mp4_isaac.convert --sku sku_2")
    return str(path)


def rough_generator(rows: int) -> TerrainGeneratorCfg:
    """Isaac Lab's rough mix at Mini Pupper scale (stand height about 7 cm, foot reach about 3 cm)."""
    return TerrainGeneratorCfg(
        size=(2.0, 2.0),
        border_width=2.0,
        num_rows=rows,
        num_cols=10,
        horizontal_scale=0.02,
        vertical_scale=0.001,
        slope_threshold=0.75,
        use_cache=False,
        curriculum=True,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25, noise_range=(0.002, 0.020), noise_step=0.002, border_width=0.10
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=0.12, grid_height_range=(0.005, 0.03), platform_width=0.5
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.005, 0.03), step_width=0.10, platform_width=0.5,
                border_width=0.10, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.15, step_height_range=(0.005, 0.03), step_width=0.10, platform_width=0.5,
                border_width=0.10, holes=False,
            ),
            "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.15, slope_range=(0.0, 0.35), platform_width=0.5, border_width=0.10
            ),
        },
    )


def terrain_levels(env, env_ids, promote_m: float, command_name: str = "twist"):
    """Up a row after walking ``promote_m`` from the origin; down after less than half the commanded distance."""
    import torch

    terrain = env.scene.terrain
    robot = env.scene["robot"]
    pos = mdp._t(robot.data.root_pos_w)[env_ids, :2]
    distance = torch.linalg.norm(pos - env.scene.env_origins[env_ids, :2], dim=1)
    command = env.command_manager.get_command(command_name)
    move_up = distance > promote_m
    move_down = (distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def _actor_terms(noisy: bool, omni: bool = True) -> dict:
    return {
        "ang_vel": ObservationTermCfg(func=mdp.obs_ang_vel, params={"noisy": noisy}),
        "gravity": ObservationTermCfg(func=mdp.obs_gravity, params={"noisy": noisy}),
        "command": ObservationTermCfg(func=mdp.obs_command, params={"omni": omni}),
        "pos": ObservationTermCfg(func=mdp.obs_pos, params={"noisy": noisy}),
        "vel": ObservationTermCfg(func=mdp.obs_vel, params={"noisy": noisy}),
        "prev_action": ObservationTermCfg(func=mdp.obs_prev_action),
    }


def _group(terms: dict, history: int = 0) -> ObservationGroupCfg:
    group = ObservationGroupCfg()
    for name, term in terms.items():
        setattr(group, name, term)
    group.concatenate_terms = True
    group.enable_corruption = False
    if history > 1:
        group.history_length = history
        group.flatten_history_dim = True
        group.history_order = "term"
    return group


def make_env_cfg(opt: TaskOptions) -> ManagerBasedRLEnvCfg:
    if opt.terrain not in TERRAINS:
        raise ValueError(f"unknown terrain {opt.terrain!r}; expected one of {TERRAINS}")
    rough = opt.terrain == "rough"

    scene = InteractiveSceneCfg(num_envs=opt.num_envs, env_spacing=1.0)
    scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator" if rough else "plane",
        terrain_generator=rough_generator(opt.rows) if rough else None,
        max_init_terrain_level=opt.max_init_level if rough else None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.45, 0.45)),
        debug_vis=False,
    )
    scene.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path(), make_uninstanceable=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, opt.spawn_z), joint_pos=mdp.stand_joint_pos(), joint_vel={".*": 0.0}
        ),
        actuators={"servos": PupperBamActuatorCfg(physics_dt=PHYSICS_DT)},
    )
    scene.contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*", history_length=3, track_air_time=True
    )
    scene.imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/" + mdp.BASE)
    scene.light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DistantLightCfg(intensity=3000.0))

    omni = opt.vy_max > 0.0 or opt.wz_max > 0.0 or opt.heading == "loop"
    critic = _actor_terms(False, omni)
    critic.update({
        "lin_vel": ObservationTermCfg(func=mdp.obs_true_lin_vel),
        "height": ObservationTermCfg(func=mdp.obs_base_height),
        "contact": ObservationTermCfg(func=mdp.obs_foot_contact),
    })
    observations = {}
    observations["actor"] = _group(_actor_terms(opt.noisy, omni), opt.history)
    observations["critic"] = _group(critic)

    actions = {}
    actions["servo"] = mdp.ServoActionCfg(scale=tuple(opt.action_scale), chain_ms=tuple(opt.chain_ms))

    fixed = opt.fixed_command
    loop = opt.heading == "loop" and not fixed
    commands = {}
    commands["twist"] = lab_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e6, 1.0e6) if fixed else tuple(opt.resample_s),
        rel_standing_envs=0.0 if fixed else opt.rel_standing,
        rel_heading_envs=opt.rel_heading if loop else 0.0,
        heading_command=loop,
        heading_control_stiffness=opt.heading_kp,
        debug_vis=False,
        ranges=lab_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(fixed[0], fixed[0]) if fixed else (0.0, opt.vx_max),
            lin_vel_y=(fixed[1], fixed[1]) if fixed else (-opt.vy_max, opt.vy_max),
            ang_vel_z=(fixed[2], fixed[2]) if fixed else (-opt.wz_max, opt.wz_max),
            heading=(-math.pi, math.pi) if loop else None,
        ),
    )

    events = {}
    events["noise"] = EventTermCfg(
        func=mdp.install_noise, mode="startup",
        params={"start": mdp.NOISE_START, "end": mdp.noise_profile(opt.noise), "ramp": opt.noise_ramp},
    )
    events["masses"] = EventTermCfg(
        func=mdp.body_masses, mode="startup",
        params={"heatsink": opt.heatsink, "base_scale": (0.9, 1.1) if opt.randomise else (1.0, 1.0),
                "com_xy": 0.005 if opt.randomise else 0.0, "com_z": 0.003 if opt.randomise else 0.0},
    )
    if opt.randomise:
        events["foot_friction"] = EventTermCfg(
            func=lab_mdp.randomize_rigid_body_material, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=mdp.FEET),
                    "static_friction_range": tuple(opt.friction), "dynamic_friction_range": tuple(opt.friction),
                    "restitution_range": (0.0, 0.0), "num_buckets": 64, "make_consistent": True},
        )
        if opt.push_mps > 0.0:
            p = opt.push_mps
            events["push"] = EventTermCfg(
                func=lab_mdp.push_by_setting_velocity, mode="interval", interval_range_s=(3.0, 6.0),
                params={"velocity_range": {"x": (-p, p), "y": (-p, p)}},
            )
    events["reset_base"] = EventTermCfg(
        func=lab_mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {} if fixed else {"yaw": (-math.pi, math.pi)}, "velocity_range": {}},
    )
    events["reset_joints"] = EventTermCfg(
        func=lab_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)},
    )

    rewards = {}
    rewards["track_lin_vel"] = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"std": 0.10})
    rewards["track_ang_vel"] = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=1.0, params={"std": 0.25})
    rewards["lin_vel_z"] = RewardTermCfg(func=lab_mdp.lin_vel_z_l2, weight=-1.0)
    rewards["ang_vel_xy"] = RewardTermCfg(func=lab_mdp.ang_vel_xy_l2, weight=-0.10)
    rewards["torques"] = RewardTermCfg(func=mdp.servo_torques_l2, weight=-1.0e-4)
    rewards["joint_acc"] = RewardTermCfg(func=lab_mdp.joint_acc_l2, weight=-1.0e-7)
    rewards["action_rate"] = RewardTermCfg(func=lab_mdp.action_rate_l2, weight=-0.02)
    rewards["upright"] = RewardTermCfg(func=lab_mdp.flat_orientation_l2, weight=-1.0)
    rewards["air_time"] = RewardTermCfg(
        func=mdp.feet_air_time, weight=0.25, params={"threshold_min": 0.05, "threshold_max": 0.30}
    )
    rewards["thigh_contact"] = RewardTermCfg(func=mdp.thigh_contacts, weight=-0.5)
    rewards["stand_still"] = RewardTermCfg(func=mdp.stand_still_l2, weight=-1.0)
    if opt.progress:
        rewards["progress"] = RewardTermCfg(func=mdp.Progress, weight=opt.progress)
    if opt.abduction:
        rewards["abduction"] = RewardTermCfg(func=mdp.abduction_l2, weight=opt.abduction)

    terminations = {}
    terminations["time_out"] = TerminationTermCfg(func=lab_mdp.time_out, time_out=True)
    terminations["tilt"] = TerminationTermCfg(func=mdp.tilt_beyond, params={"limit_rad": opt.tilt_limit})
    terminations["base_contact"] = TerminationTermCfg(func=mdp.base_contact)

    curriculum = {}
    if opt.noisy and opt.noise_ramp:
        curriculum["noise_ramp"] = CurriculumTermCfg(
            func=mdp.ramp_noise, params={"hold_iters": opt.noise_hold_iters, "ramp_iters": opt.noise_ramp_iters}
        )
    if rough and opt.rows > 1:
        curriculum["terrain_levels"] = CurriculumTermCfg(func=terrain_levels, params={"promote_m": 0.8})

    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=opt.njmax, nconmax=opt.nconmax, iterations=opt.solver_iterations,
            ls_iterations=opt.solver_ls_iterations, integrator="implicitfast", cone="pyramidal",
        ),
    )
    cfg = ManagerBasedRLEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        sim=sim_utils.SimulationCfg(dt=PHYSICS_DT, physics=physics, use_newton_actuators=False),
        decimation=DECIMATION,
        episode_length_s=opt.episode_s,
    )
    cfg.sim.render_interval = DECIMATION
    cfg.scene.contact_forces.update_period = PHYSICS_DT
    return cfg
