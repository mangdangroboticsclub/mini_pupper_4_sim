"""The SKU2 forward-walking task in mjlab: the base the terrain task (``terrain_task``) builds on.

One policy step is 20 ms (50 Hz) of twenty 1 ms physics steps. Commands: forward speed U(0.05, 0.20) m/s per episode
(the firmware trot's designed range), no sideways or turning command. Episodes: 10 s from the stand, deterministic
resets. Rewards: tracking, the regularisers, straightness, swing progress with a recent touchdown, heavy stance slip.
Randomisation: the feet's friction, the base COM, the per-episode chain delay, the observation noise (ramped).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .. import paths
from ..actuators import arms as arms_mod
from ..actuators.fits import PHYSICS_DT
from ..robot.sku2 import BODY_CONTACT_BODIES, FOOT_GEOMS, robot_cfg
from . import mdp

DECIMATION = 20
EPISODE_S = 10.0
VX_RANGE = (0.05, 0.20)
CHAIN_MS = (8.0, 19.0)
# Observation noise, uniform half-widths (rad/s, unit vector, rad, rad/s). The ramp starts from a clean floor and ends
# at a measured profile: noise_n1.json (one half-width per channel, the variance excess of SKU2's logs over the
# simulation along a trot) or noise_n2.json (one per element, the sensor noise plus the broadband excess in motion; the
# training default, ``python -m mp4_sim.gpu.train --noise``). NOISE_END is n1.
NOISE_START = {"ang_vel": 0.02, "gravity": 0.01, "pos": 0.002, "vel": 0.05}
NOISE_END = {k: float(v) for k, v in json.loads(paths.NOISE_N1_JSON.read_text())["profile_uniform_half_width"].items()}
NOISE_HOLD_ITERS, NOISE_RAMP_ITERS = 150, 300


def noise_profile(name: str) -> dict:
    """The end amplitudes by name: ``n1`` (one half-width per channel), ``n2`` (one per element, the training
    default) and ``n2x2`` (twice n2, for robustness evaluation)."""
    if name == "n1":
        return dict(NOISE_END)
    if name in ("n2", "n2x2"):
        prof = json.loads(paths.NOISE_N2_JSON.read_text())["profile_uniform_half_width"]
        k = 2.0 if name == "n2x2" else 1.0
        return {ch: [k * float(v) for v in vals] for ch, vals in prof.items()}
    raise ValueError(f"unknown noise profile {name!r}")


@dataclass
class TaskOptions:
    arm: str = "stock"
    num_envs: int = 4096
    noise_start: dict = field(default_factory=lambda: dict(NOISE_START))
    noise_end: dict = field(default_factory=lambda: dict(NOISE_END))
    noise_ramp: bool = True                 # False: the end amplitudes from the start (evaluation under noise)
    noisy: bool = True                      # False: clean observations (evaluation, play)
    randomise: bool = True                  # friction, COM, chain delay drawn per env
    chain_ms: tuple = CHAIN_MS
    vx_range: tuple = VX_RANGE
    action_scale: tuple = (0.10, 0.40, 0.42) * 4
    episode_s: float = EPISODE_S
    visuals: bool = False                   # keep the meshes (rendering only)
    fixed_command: tuple | None = None      # (vx, vy, wz) for every env (evaluation)
    resample_s: tuple = (1.0e6, 1.0e6)      # command resampling interval (the forward task: never)


def install_noise(env, env_ids, start: dict, end: dict, ramp: bool):
    if not hasattr(env, "sim_noise"):
        env.sim_noise = mdp.NoiseTable(env, start, end)
        if not ramp:
            env.sim_noise.amp = dict(env.sim_noise.end)


def _obs_terms(noisy: bool) -> dict:
    return {
        "ang_vel": ObservationTermCfg(func=mdp.obs_ang_vel, params={"noisy": noisy}),
        "gravity": ObservationTermCfg(func=mdp.obs_gravity, params={"noisy": noisy}),
        "command": ObservationTermCfg(func=mdp.obs_command, params={"omni": False}),
        "pos": ObservationTermCfg(func=mdp.obs_pos, params={"noisy": noisy}),
        "vel": ObservationTermCfg(func=mdp.obs_vel, params={"noisy": noisy}),
        "prev_action": ObservationTermCfg(func=mdp.obs_prev_action),
    }


def make_env_cfg(opt: TaskOptions | None = None) -> ManagerBasedRlEnvCfg:
    opt = opt or TaskOptions()
    arm = arms_mod.get(opt.arm)
    critic = _obs_terms(False)
    critic.update({
        "lin_vel": ObservationTermCfg(func=mdp.obs_true_lin_vel),
        "height": ObservationTermCfg(func=mdp.obs_base_height),
        "contact": ObservationTermCfg(func=mdp.obs_foot_contact),
    })
    observations = {
        "actor": ObservationGroupCfg(terms=_obs_terms(opt.noisy), concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
        "anchor": ObservationGroupCfg(terms=_obs_terms(False), concatenate_terms=True, enable_corruption=False),
    }
    sensors = (
        ContactSensorCfg(name="feet_contact",
                         primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found", "force"), reduce="netforce", num_slots=1, track_air_time=True),
        ContactSensorCfg(name="body_contact",
                         primary=ContactMatch(mode="body", pattern=BODY_CONTACT_BODIES, entity="robot"),
                         secondary=ContactMatch(mode="body", pattern="terrain"),
                         fields=("found",), reduce="none", num_slots=1),
    )
    events = {
        "noise": EventTermCfg(func=install_noise, mode="startup",
                              params={"start": opt.noise_start, "end": opt.noise_end, "ramp": opt.noise_ramp}),
        "reset_base": EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                   params={"pose_range": {}, "velocity_range": {}}),
        "reset_joints": EventTermCfg(func=vel_mdp.reset_joints_by_offset, mode="reset",
                                     params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                             "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "command_delay": EventTermCfg(func=mdp.reset_command_delay, mode="reset",
                                      params={"latency_steps": arm.latency_steps,
                                              "chain_ms": opt.chain_ms if opt.randomise else
                                              (sum(opt.chain_ms) / 2,) * 2}),
    }
    if arm.name.startswith("bam"):
        from ..actuators.bam_servo import bam_servo_init
        events["bam_init"] = EventTermCfg(func=bam_servo_init, mode="startup")
    if opt.randomise:
        events["foot_friction"] = EventTermCfg(
            func=envs_mdp.dr.geom_friction, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
                    "axes": [0], "ranges": (0.4, 1.2), "shared_random": True})
        events["base_com"] = EventTermCfg(
            func=envs_mdp.dr.body_com_offset, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "add",
                    "ranges": {0: (-0.005, 0.005), 1: (-0.005, 0.005), 2: (-0.003, 0.003)}})
    rewards = {
        "track_lin_vel": RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"std": 0.10}),
        "track_ang_vel": RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=0.50, params={"std": 0.10}),
        "lin_vel_z": RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-1.0),
        "ang_vel_xy": RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.10),
        "torques": RewardTermCfg(func=mdp.torques_l2, weight=-1.0e-4),
        "joint_acc": RewardTermCfg(func=mdp.joint_acc_l2, weight=-1.0e-7),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
        "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-1.0),
        "yaw": RewardTermCfg(func=mdp.yaw_l2, weight=-0.40),
        "lateral_position": RewardTermCfg(func=mdp.lateral_position_l2, weight=-1.50),
        "lateral_velocity": RewardTermCfg(func=mdp.lateral_velocity_l2, weight=-0.40),
        "swing_progress": RewardTermCfg(func=mdp.SwingProgressTouchdown, weight=0.80, params={}),
        "stance_slip": RewardTermCfg(func=mdp.stance_foot_slip_l2, weight=-2.0, params={"threshold": 1.0}),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "tilt": TerminationTermCfg(func=mdp.tilt_beyond, params={"limit_rad": 0.7}),
        "body_contact": TerminationTermCfg(func=mdp.body_contact),
    }
    curriculum = {}
    if opt.noise_ramp and opt.noisy:
        curriculum["noise_ramp"] = CurriculumTermCfg(
            func=mdp.ramp_noise, params={"hold_iters": NOISE_HOLD_ITERS, "ramp_iters": NOISE_RAMP_ITERS})
    commands = {"twist": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False,
        rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=tuple(opt.vx_range), lin_vel_y=(0.0, 0.0),
                                                 ang_vel_z=(0.0, 0.0)))}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=opt.num_envs, env_spacing=1.0,
                       entities={"robot": robot_cfg(arm.actuators, strip_visuals=not opt.visuals)}, sensors=sensors),
        observations=observations,
        actions={"servo": mdp.ServoActionCfg(scale=tuple(opt.action_scale))},
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base",
                            distance=0.6, elevation=-10.0, azimuth=90.0),
        sim=SimulationCfg(njmax=300, nconmax=None,
                          mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20)),
        decimation=DECIMATION,
        episode_length_s=opt.episode_s,
    )


def describe(opt: TaskOptions) -> dict:
    """Everything a run's saved config must name (the actuator model's parameters, the sources)."""
    from ..robot.sku2 import sources
    arm = arms_mod.get(opt.arm)
    return {"options": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.__dict__.items()},
            "arm": arm.name, "arm_params": arm.params, "latency_steps": list(arm.latency_steps),
            "decimation": DECIMATION, "physics_dt": PHYSICS_DT, "sources": sources(getattr(opt, "masses", None))}
