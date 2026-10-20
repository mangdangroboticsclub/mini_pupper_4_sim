"""The SKU2 terrain task: the forward-walking task's robot, actuators, observations, noise and delays (``env_cfg``), on a
terrain curriculum (the presets in ``terrain.PRESETS``, ``rough_basic`` by default), commanded by a forward speed and a
yaw rate.

Built on ``env_cfg.make_env_cfg``, with:
- **terrain:** mjlab's generator with our sub-terrains, a curriculum on the forward progress walked
  (``terrain_levels_path``, guarded against the first reset) and our own bounds termination;
- **commands:** vx in [0, ``vx_max``], wz in ±``wz_max``, vy 0, a share standing, resampled every 5-10 s; observed
  whole (the omni observation form, 45 floats). With ``heading="loop"`` most envs get their wz from a heading loop, as
  on the robot: a target heading relative to the current one, wz = clip(kp · error) (``heading_law="pi"``: the
  robot's PI law, ``heading.pi_step``);
- **rewards:** body-frame velocity and yaw-rate tracking, the heading against the integrated yaw-rate command
  (``HeadingTrack``, with ``heading="integral"`` only: privileged, read by the reward only), optional alive and
  termination terms, uprightness relative to the terrain normal, foot clearance
  over the terrain under each foot, air time, stance slip, leg-terrain contacts;
- **critic:** the forward critic plus a height scan around the body and each foot's height over the terrain;
- **robot:** shin-base and foot-base collisions (the model has none) and the 478 g mass file by default;
- 20 s episodes, a random yaw at reset, small pushes, mass randomisation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.builtin_sensor import ObjRef
from mjlab.sensor.raycast_sensor import GridPatternCfg, RayCastSensorCfg, RingPatternCfg
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from ..actuators import arms as arms_mod
from ..contract import ABDUCTION
from ..robot.sku2 import FOOT_GEOMS, FOOT_SITES, LEG_COLLISION_GEOMS, robot_cfg
from . import mdp
from . import terrain as terrain_mod
from .env_cfg import TaskOptions, make_env_cfg
from .mdp import _foot_site_ids

FOOT_R = 0.0049


def _fresh_raycasts() -> None:
    """mjlab 1.3.0 caches a sensor's data until ``reset`` or ``update``, but raycasts are refreshed later, in
    ``sense()`` (after the rewards): a reward that reads a raycast sensor caches the previous step's rays and the
    observations then get them too (mjlab issue #998, fixed in later mjlab releases). Invalidate the cache when ``sense()`` writes
    new rays. Rewards still run before ``sense()``, so reward terms here take the moving part (the foot) from the
    current state and only the terrain from the rays."""
    from mjlab.sensor.raycast_sensor import RayCastSensor
    if getattr(RayCastSensor, "_sim_fresh", False):
        return
    original = RayCastSensor.postprocess_rays

    def postprocess_rays(self):
        original(self)
        self._invalidate_cache()

    RayCastSensor.postprocess_rays = postprocess_rays
    RayCastSensor._sim_fresh = True


_fresh_raycasts()
TERRAIN_ACTION_SCALE = (0.25, 0.50, 0.50) * 4
BORDER = 0.5


TERRAIN_PRESETS = terrain_mod.PRESETS
TERRAINS = (*TERRAIN_PRESETS, "stairs_built", "flat")
# "stairs_built": a straight flight from these parameters (m), the default a small built staircase
STAIR_BUILT = {"risers": (0.025,), "tread": 0.05, "steps": 4, "width": 0.20, "landing": 0.25}


@dataclass
class TerrainOptions(TaskOptions):
    terrain: str = "rough_basic"            # a TERRAINS name
    rows: int = 10                          # difficulty levels
    tile: float = terrain_mod.TILE_SIZE
    terrain_seed: int = 0
    max_init_level: int = 2
    narrow_risers: tuple | None = None      # --terrain stairs_narrow: the flights' nominal riser range (m); None = (0.010, 0.035)
    vx_max: float = 0.15
    wz_max: float = 0.6
    rel_standing: float = 0.1
    resample_s: tuple = (5.0, 10.0)
    action_scale: tuple = TERRAIN_ACTION_SCALE
    episode_s: float = 20.0
    history: int = 0                        # actor observation history (0: the 45-float contract)
    masses: str | None = "sku2_masses_478g.json"        # 478 g, SKU2 with a Pi heatsink; None: sku2_masses.json (450 g)
    self_collision: bool = True             # shins and feet collide with the base
    friction_range: tuple = (0.5, 1.3)
    # the feet's contact (timeconst, dampratio). MuJoCo's (0.02, 1) lets a walking foot sink 3.3 mm (p99); (0.01, 1)
    # 1.7 mm, (0.005, 1) 0.6 mm, the flat policy walking the same
    foot_solref: tuple | None = (0.01, 1.0)
    push_mps: float = 0.05
    fixed_command: tuple | None = None      # (vx, 0, wz) for every env (evaluation)
    level: int | None = None                # evaluation: every env on this difficulty row
    # "integral": wz sampled, HeadingTrack penalises the heading against the integrated wz (unbounded, unobserved);
    # "loop": wz from a heading loop on a target relative to the current heading (the robot's outer loop), no
    # HeadingTrack; the rest of the envs keep a sampled wz
    heading: str = "integral"
    heading_kp: float = 2.0                 # heading.KP
    heading_turn: float = 1.0               # rad: new targets within ± this of the current heading
    rel_heading: float = 0.75
    alive: float = 0.0                      # reward weight per second upright (mjlab is_alive)
    terminated: float = 0.0                 # reward weight on a termination that is not a time-out (is_terminated);
                                            # mjlab scales rewards by dt (0.02 s): -250 is -5 per fall
    progress: float = 0.0                   # weight per second of progress_linear (1 = walking at the command)
    stumble: float = 0.0                    # weight per second per foot pushed mostly sideways (feet_stumble)
    clearance: str = "speed"                # "speed": foot_clearance_now; "apex": SwingApexClearance
    clearance_weight: float = -4.0          # "apex": per metre of apex deficit, per touchdown
    clearance_target: float = 0.025         # "apex": the swing apex wanted above the terrain under and ahead of the foot
    foot_softness: tuple | None = None      # the feet's solref timeconst drawn per env (s), e.g. (0.005, 0.02)
    abduction: float = 0.0                  # weight per second of abduction_deviation_l2 (keeps the stance)
    leg_contact: float = -0.5               # weight per second per thigh or shin capsule touching the terrain
    stair_risers: tuple = STAIR_BUILT["risers"]     # "stairs_built": one flight per riser (m), one column each
    stair_tread: float = STAIR_BUILT["tread"]       # "stairs_built": the tread between risers (m)
    stair_steps: int = STAIR_BUILT["steps"]         # "stairs_built": risers per flight
    stair_width: float | None = None        # "stairs_built": the treads' width (m; None: the whole tile)
    stair_landing: float | None = None      # "stairs_built": the top tread's length (m; None: to the tile's edge)
    tilt_limit: float = 0.7                 # rad: the body's tilt from vertical that ends an episode as a fall
    stair_exit: bool = False                # on jittered-stair tiles: end the episode once off the stairs (a success),
                                            # promote on that, demote on a fall or never leaving the platform
    heading_law: str = "p"                  # the simulated heading loop: "p" mjlab's clip(kp err, ±wz_max); "pi" the
                                            # robot's (heading.pi_step: PI, anti-windup, ±heading_wz)
    heading_wz: float = 0.4                 # "pi": the loop's output limit (rad/s; heading.WZ_MAX)
    cross_track: float = 0.0                # weight per second of the clipped cross-track drift (CrossTrackClipped)
    cross_track_clip: float = 0.10          # m: the drift at which that term saturates (cost = weight)
    yaw_catch: tuple | None = None          # (lo, hi) N m: yaw torque pulses on the base (YawCatch), random sign
    yaw_catch_start: int = 0                # env steps before the pulses start (iterations x 24)
    heading_reward: float = 0.0             # weight per second of heading_align, exp(-(error / 0.25)^2), on loop envs


# ---------------------------------------------------------------------------------------------------- terms
class PathTracker(ManagerTermBase):
    """Accumulates the forward progress walked (the body-frame forward speed, positive part, integrated: a turning
    episode is not penalised, body sway and stepping in place add nothing) and the distance commanded, for the
    curriculum. Returns 0: it is a reward term only so that it runs every step."""

    def __init__(self, cfg, env):
        super().__init__(env)
        env.sim_path = torch.zeros(env.num_envs, device=env.device)
        env.sim_cmd_path = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._env.sim_path[:] = 0
            self._env.sim_cmd_path[:] = 0
        else:
            self._env.sim_path[env_ids] = 0
            self._env.sim_cmd_path[env_ids] = 0

    def __call__(self, env, command_name: str = "twist"):
        vx = env.scene["robot"].data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
        c = env.command_manager.get_command(command_name)[:, :2].norm(dim=-1)
        env.sim_path += vx * env.step_dt
        env.sim_cmd_path += c * env.step_dt
        return torch.zeros_like(vx)


class RelativeHeadingCommand(UniformVelocityCommand):
    """mjlab's heading command with each new target drawn relative to the current heading (mjlab draws it in the
    world frame, which with a random spawn yaw means long saturated turns). The offset is applied at the next update:
    at a reset the body poses (and so ``heading_w``) are stale until the next forward pass.

    ``heading_law`` "pi": the heading envs' wz from the robot's loop (``heading.pi_step``, its integrator reset with
    each new target) instead of mjlab's P-only law. On the columns ``narrow_cols`` (narrow flights) every env is a
    heading env walking forward at U(``narrow_vx``), never standing, and holds the heading it spawned with for the
    whole episode (as the robot's loop holds its start heading): later resamples keep the target."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._offset = torch.zeros(self.num_envs, device=self.device)
        self._pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._integ = torch.zeros(self.num_envs, device=self.device)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _narrow(self, env_ids):
        cols = self.cfg.narrow_cols
        t = getattr(self._env.scene, "terrain", None)
        if not cols or t is None or getattr(t, "terrain_types", None) is None:
            return torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        return torch.isin(t.terrain_types[env_ids], torch.tensor(cols, device=self.device))

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._fresh[ids] = True
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        keep = self.heading_target[env_ids].clone()
        keep_i = self._integ[env_ids].clone()
        super()._resample_command(env_ids)
        if self.cfg.heading_command:
            self._offset[env_ids] = self.heading_target[env_ids]
            self._pending[env_ids] = True
            self._integ[env_ids] = 0.0
            nar = self._narrow(env_ids)
            if bool(nar.any()):
                ids = env_ids[nar]
                fresh = self._fresh[ids]
                self.is_heading_env[ids] = True
                self.is_standing_env[ids] = False
                self.vel_command_b[ids, 0] = torch.empty(len(ids), device=self.device).uniform_(*self.cfg.narrow_vx)
                self.vel_command_b[ids, 1] = 0.0
                self._offset[ids] = 0.0                              # a fresh episode: hold the spawn heading
                self._pending[ids] = fresh                            # a later resample: keep the target
                self.heading_target[ids] = torch.where(fresh, self.heading_target[ids], keep[nar])
                self._integ[ids] = torch.where(fresh, 0.0, keep_i[nar])       # mid-episode: the loop runs on
        self._fresh[env_ids] = False

    def _update_command(self):
        if self.cfg.heading_command and bool(self._pending.any()):
            from mjlab.utils.lab_api.math import wrap_to_pi
            ids = self._pending.nonzero(as_tuple=False).flatten()
            self.heading_target[ids] = wrap_to_pi(self.robot.data.heading_w[ids] + self._offset[ids])
            self._pending[ids] = False
        if self.cfg.heading_command and self.cfg.heading_law == "pi":
            from mjlab.utils.lab_api.math import wrap_to_pi

            from ..heading import pi_step
            err = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
            wz, integ = pi_step(err, self._integ, self._env.step_dt, wz_max=self.cfg.heading_wz)
            h = self.is_heading_env
            self._integ = torch.where(h, integ, self._integ)
            super()._update_command()                                # mjlab's law, world envs, standing
            self.vel_command_b[:, 2] = torch.where(h & ~self.is_standing_env, wz, self.vel_command_b[:, 2])
            return
        super()._update_command()


@dataclass(kw_only=True)
class RelativeHeadingCommandCfg(UniformVelocityCommandCfg):
    heading_law: str = "p"
    heading_wz: float = 0.4
    narrow_cols: tuple = ()
    narrow_vx: tuple = (0.06, 0.12)

    def build(self, env):
        return RelativeHeadingCommand(self, env)


STAIR_EXIT_MARGIN = 0.08      # m: the base this far past the stairs' last riser (Chebyshev), so the rear feet are off too
STAIR_STAYED_M = 0.30         # m: a stair episode whose base never got this far (Chebyshev) is demoted


def stair_exit(env, subs: tuple, tile: float):
    """True on a jittered-stair tile once the base is off the stairs: its Chebyshev distance from the spawn (the rings
    are square) at least the outermost riser's half-size plus ``STAIR_EXIT_MARGIN`` (capped 2 cm inside the tile).
    A time-out-like termination: the episode ends as a success before the robot meets the neighbouring tile's rim (a
    wall or a drop, which would end it as a fall). Keeps each env's furthest Chebyshev distance this episode in
    ``env.sim_cheb_max`` and the exit flags in ``env.sim_stair_exit`` for ``terrain_levels_path``."""
    t = env.scene.terrain
    n = env.num_envs
    if not hasattr(env, "_sim_extent"):
        rows = t.terrain_origins.shape[0]
        ext = torch.full((rows, len(subs)), float("inf"), device=env.device)
        for col, cfg in enumerate(subs):
            if isinstance(cfg, terrain_mod.JitterPyramidStairsCfg):
                e = torch.tensor(cfg.extents[-rows:], device=env.device)
                ext[:, col] = torch.clamp(e + STAIR_EXIT_MARGIN, max=tile / 2 - 0.02)
        env._sim_extent = ext
        _narrow_tables(env, subs)
        env._sim_stairs = torch.isfinite(ext[0]) | env._sim_narrow
        env.sim_cheb_max = torch.zeros(n, device=env.device)
        env.sim_stair_exit = torch.zeros(n, dtype=torch.bool, device=env.device)
    p = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    cheb = p.abs().max(-1).values
    new = env.episode_length_buf <= 1
    env.sim_cheb_max = torch.where(new, cheb, torch.maximum(env.sim_cheb_max, cheb))
    env.sim_stair_exit = cheb >= env._sim_extent[t.terrain_levels, t.terrain_types]
    if bool(env._sim_narrow.any()):     # a narrow flight: all four feet on its landing
        on = env._sim_narrow[t.terrain_types]
        rear = (env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 0]
                - env.scene.env_origins[:, :1]).min(-1).values
        landed = rear > env._sim_flight[t.terrain_levels, t.terrain_types, 1] + STAIR_FOOT_PAST
        env.sim_stair_exit = torch.where(on, landed, env.sim_stair_exit)
    return env.sim_stair_exit


STAIR_FOOT_PAST = 0.01          # m: a foot centre this far past a riser counts as over it (as gpu/evaluate.py)


def _narrow_tables(env, subs) -> None:
    """Per (row, column) of a ``NarrowFlightCfg`` column: the first and last riser's x and the half-width, relative to
    the spawn (``env._sim_flight``, (rows, cols, 3)); ``env._sim_narrow`` marks the columns."""
    if hasattr(env, "_sim_narrow"):
        return
    rows = env.scene.terrain.terrain_origins.shape[0]
    tab = torch.zeros(rows, len(subs), 3, device=env.device)
    nar = torch.zeros(len(subs), dtype=torch.bool, device=env.device)
    for col, cfg in enumerate(subs):
        if isinstance(cfg, terrain_mod.NarrowFlightCfg):
            f = cfg.flights[-rows:]
            tab[:, col] = torch.tensor([[x["x_first"], x["x_last"], x["width"] / 2] for x in f], device=env.device)
            nar[col] = True
    env._sim_flight, env._sim_narrow = tab, nar


def side_exit(env, subs: tuple):
    """True on a narrow flight once a stance foot (on a surface: its centre within its radius + 2 mm of the terrain
    under it) past the first riser, or the base, is beyond the flight's half-width from its centre line: a foot off
    a side, which counts as a failure on a real staircase (a termination, not a time-out)."""
    _narrow_tables(env, subs)
    if not bool(env._sim_narrow.any()):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    t = env.scene.terrain
    on = env._sim_narrow[t.terrain_types]
    tab = env._sim_flight[t.terrain_levels, t.terrain_types]
    o = env.scene.env_origins
    fp = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), :2] - o[:, None, :2]
    stance = (fp[..., 0] > tab[:, :1] + STAIR_FOOT_PAST) & (foot_heights_centre(env) <= FOOT_R + 0.002)
    off = (stance & (fp[..., 1].abs() > tab[:, 2:3])).any(-1)
    base_y = (env.scene["robot"].data.root_link_pos_w[:, 1] - o[:, 1]).abs()
    return on & (off | (base_y > tab[:, 2]))


def reset_base_narrow(env, env_ids, pose_range: dict, narrow_pose_range: dict, narrow_cols: tuple):
    """The base reset: ``pose_range`` everywhere, then on the narrow flights' columns ``narrow_pose_range`` instead
    (facing the flight within a few degrees, a few mm off its line: the placement tolerance)."""
    vel_mdp.reset_root_state_uniform(env, env_ids, pose_range=pose_range, velocity_range={})
    t = env.scene.terrain
    if env_ids is None or not narrow_cols or t is None:
        return
    nar = torch.isin(t.terrain_types[env_ids], torch.tensor(narrow_cols, device=env.device))
    if bool(nar.any()):
        vel_mdp.reset_root_state_uniform(env, env_ids[nar], pose_range=narrow_pose_range, velocity_range={})


def terrain_levels_path(env, env_ids, promote_m: float, min_cmd_m: float = 0.3):
    """Rudin's rule on the forward progress, not the displacement (turning episodes curl): up a level when the episode walked at
    least ``promote_m`` and ended without falling; down when it walked less than half the commanded path or fell.
    Episodes with under ``min_cmd_m`` commanded (standing; the very first reset) do not move."""
    t = env.scene.terrain
    if t is None or getattr(t, "terrain_levels", None) is None or not hasattr(env, "sim_path"):
        return None
    path, cmd = env.sim_path[env_ids], env.sim_cmd_path[env_ids]
    fell = env.termination_manager.terminated[env_ids]
    judged = cmd >= min_cmd_m
    up = judged & ~fell & (path >= promote_m)
    down = judged & ~up & (fell | (path < 0.5 * cmd))
    if hasattr(env, "sim_stair_exit"):   # stair tiles (stair_exit): up only off the stairs; down on a fall or staying
        st = env._sim_stairs[t.terrain_types[env_ids]]
        out = env.sim_stair_exit[env_ids]
        s_up = ~fell & out
        s_down = ~out & (fell | (judged & (env.sim_cheb_max[env_ids] < STAIR_STAYED_M)))
        up, down = torch.where(st, s_up, up), torch.where(st, s_down, down)
    t.update_env_origins(env_ids, up, down)
    return torch.mean(t.terrain_levels.float())


def family_levels(env, env_ids, names: tuple):
    """The mean terrain level per family (column), logged as ``Curriculum/family_levels/<name>``: the mean over all
    families can hide one family's collapse. No side effect."""
    t = env.scene.terrain
    if t is None or getattr(t, "terrain_levels", None) is None:
        return None
    lv, ty = t.terrain_levels.float(), t.terrain_types
    return {n: lv[ty == i].mean() for i, n in enumerate(names) if bool((ty == i).any())}


def progress_linear(env, command_name: str = "twist", min_cmd: float = 0.02):
    """Body speed along the commanded direction over the commanded speed, clipped to [0, 1]; 0 below ``min_cmd``.
    Linear, so any step towards the command pays and standing pays nothing (velocity tracking's exp kernel pays
    standing about half at these speeds)."""
    c = env.command_manager.get_command(command_name)[:, :2]
    speed = c.norm(dim=-1)
    v = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    along = (v * c).sum(-1) / speed.clamp(min=min_cmd)
    return (along / speed.clamp(min=min_cmd)).clamp(0.0, 1.0) * (speed >= min_cmd).float()


def feet_stumble(env, ratio: float = 1.5, sensor_name: str = "feet_contact"):
    """Feet in contact with a mostly horizontal net force, |F_xy| > ratio · |F_z|: a foot pushed against a step's face
    (Rudin et al. 2022). The sensor's net force is in the world frame."""
    d = env.scene[sensor_name].data
    n = d.found.shape[1]
    f = d.force.view(d.force.shape[0], n, 3)
    hit = (d.found > 0) & (f[..., :2].norm(dim=-1) > ratio * f[..., 2].abs())
    return hit.float().sum(-1)


@dataclass
class AheadPatternCfg(GridPatternCfg):
    """A grid of rays shifted by ``offset`` in the frame (yaw-aligned: forward along the body's heading); started high
    so that a foot pressed against a face does not start its rays inside it."""
    offset: tuple = (0.015, 0.0, 0.05)

    def generate_rays(self, mj_model, device):
        off, dirs = super().generate_rays(mj_model, device)
        return off + torch.tensor(self.offset, device=device, dtype=off.dtype), dirs


def ground_ahead(env, ahead: str = "foot_ahead", under: str = "foot_height") -> torch.Tensor:
    """The highest terrain under each foot or up to 3 cm ahead of it (world z), from the rays of the last sense()."""
    da, du = env.scene[ahead].data, env.scene[under].data
    B, F = du.heights.shape[0], du.heights.shape[1]
    za = torch.where(da.distances.view(B, F, -1) >= 0, da.hit_pos_w[..., 2].view(B, F, -1),
                     torch.full_like(da.distances.view(B, F, -1), -1.0e3)).amax(-1)
    zu = torch.where(du.distances.view(B, F, -1) >= 0, du.hit_pos_w[..., 2].view(B, F, -1),
                     torch.full_like(du.distances.view(B, F, -1), -1.0e3)).amax(-1)
    return torch.maximum(za, zu)


class SwingApexClearance(ManagerTermBase):
    """One-sided swing height: at each touchdown, how far the swing's highest clearance stayed below ``target``; the
    clearance is the foot sphere's bottom above the highest terrain under the foot or 3 cm ahead of it. Only swings
    of at least ``min_air`` count: on rough ground the contact flickers during stance, and each flicker would
    otherwise "land" with no lift and pay the whole target. Returns metres per touchdown / dt, so the weight is the
    penalty per metre of deficit per touchdown. Only while commanded to move."""

    def __init__(self, cfg, env):
        super().__init__(env)
        self._apex = torch.zeros(env.num_envs, 4, device=env.device)
        self._air = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)
        self._air_s = torch.zeros(env.num_envs, 4, device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._apex[ids] = 0.0
        self._air[ids] = False
        self._air_s[ids] = 0.0

    def __call__(self, env, target: float, min_air: float = 0.06, command_name: str = "twist",
                 command_threshold: float = 0.02):
        found = env.scene["feet_contact"].data.found > 0
        z = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 2]
        clear = z - FOOT_R - ground_ahead(env)
        air = ~found
        self._apex = torch.where(air & self._air, torch.maximum(self._apex, clear), torch.where(air, clear, self._apex))
        touchdown = self._air & found & (self._air_s >= min_air - 1e-6)
        deficit = (target - self._apex).clamp(0.0, target) * touchdown.float()
        self._air_s = torch.where(air, self._air_s + env.step_dt, torch.zeros_like(self._air_s))
        self._air = air
        cmd = env.command_manager.get_command(command_name)
        active = (cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()) > command_threshold
        return deficit.sum(-1) * active.float() / env.step_dt


@requires_model_fields("geom_solref")
def foot_softness(env, env_ids, ranges, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The feet's contact timeconst (solref[0]) drawn per env, the same for all four; the feet have contact priority,
    so theirs is the one used against the terrain."""
    from mjlab.envs.mdp.dr._core import _randomize_model_field
    _randomize_model_field(env, env_ids, "geom_solref", entity_type="geom", ranges=ranges, operation="abs",
                           asset_cfg=asset_cfg, axes=[0], shared_random=True, default_axes=[0], valid_axes=[0, 1])


class CrossTrackClipped(mdp.CrossTrack):
    """``mdp.CrossTrack`` (the drift across the commanded direction, along-track lag forgiven) for the terrain task:
    the cost is |drift| / ``clip``, saturating at 1 (a bounded term: an unbounded square at the drifts the policy
    starts with would swamp the return); a standing command costs nothing and re-anchors on the body every step. The
    body-frame drift is left in ``env.sim_cross_track`` for the critic (``mdp.obs_track_errors``)."""

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self._env, "sim_cross_track"):       # the critic's first observation of a new episode
            ids = slice(None) if env_ids is None else env_ids
            self._env.sim_cross_track[ids] = 0.0

    def __call__(self, env, command_name="twist", clip: float = 0.10):
        e2 = super().__call__(env, command_name)
        cmd = env.command_manager.get_command(command_name)[:, :2].norm(dim=-1)
        moving = cmd > 0.02
        p = env.scene["robot"].data.root_link_pos_w[:, :2]
        self._target = torch.where(moving[:, None], self._target, p)
        env.sim_cross_track = torch.where(moving[:, None], env.sim_cross_track, torch.zeros_like(env.sim_cross_track))
        return torch.where(moving, (e2.sqrt() / clip).clamp(max=1.0), torch.zeros_like(e2))


class YawCatch(envs_mdp.apply_body_impulse):
    """A foot catching a step edge, as a disturbance: a yaw torque of U(``torque``) N m, random sign, on the base for
    U(``duration_s``), every U(``cooldown_s``) per env, from env step ``start_step`` on (mjlab's ``apply_body_impulse``
    draws every torque axis: roll and pitch at this size would tip the body rather than turn it). Use with
    mode="step"."""

    def __call__(self, env, env_ids, torque: tuple, duration_s: tuple, cooldown_s: tuple, asset_cfg,
                 start_step: int = 0) -> None:
        if env.common_step_counter < start_step:
            return
        dt = self._step_dt
        self._time_remaining[self._active] -= dt
        expired = self._active & (self._time_remaining <= 0)
        if bool(expired.any()):
            ids = expired.nonzero(as_tuple=False).squeeze(-1)
            z = torch.zeros((len(ids), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(z, z, env_ids=ids, body_ids=self._body_ids)
            self._active[ids] = False
            self._interval_time_left[ids] = torch.empty(len(ids), device=self._device).uniform_(*cooldown_s)
        self._interval_time_left -= dt
        go = (~self._active) & (self._interval_time_left <= 0)
        if not bool(go.any()):
            return
        ids = go.nonzero(as_tuple=False).squeeze(-1)
        n = len(ids)
        f = torch.zeros((n, self._num_bodies, 3), device=self._device)
        t = torch.zeros_like(f)
        mag = torch.empty(n, device=self._device).uniform_(*torque)
        sign = torch.where(torch.rand(n, device=self._device) < 0.5, -1.0, 1.0)
        t[:, :, 2] = (sign * mag)[:, None]
        self._asset.write_external_wrench_to_sim(f, t, env_ids=ids, body_ids=self._body_ids)
        self._time_remaining[ids] = torch.empty(n, device=self._device).uniform_(*duration_s)
        self._active[ids] = True
        self._interval_time_left[ids] = torch.empty(n, device=self._device).uniform_(*cooldown_s)


def out_of_bounds(env, half_x: float, half_y: float, margin: float = 0.3):
    p = env.scene["robot"].data.root_link_pos_w
    return (p[:, 0].abs() > half_x - margin) | (p[:, 1].abs() > half_y - margin)


def terrain_normal(env, sensor_name: str = "body_scan") -> torch.Tensor:
    """The mean of the scan's hit normals (world), unit length; straight up where nothing was hit or not finite."""
    d = env.scene[sensor_name].data
    n = d.normals_w                                            # (B, R, 3)
    hit = (d.distances >= 0) & torch.isfinite(n).all(-1)
    n = torch.where(hit.unsqueeze(-1), n, torch.zeros_like(n)).sum(1)
    up = torch.tensor([0.0, 0.0, 1.0], device=n.device).expand_as(n)
    norm = n.norm(dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, n / norm.clamp(min=1e-6), up)


def upright_terrain_l2(env, sensor_name: str = "body_scan"):
    """The terrain normal's tilt in the body frame (0 when the body is parallel to the local ground)."""
    from mjlab.utils.lab_api.math import quat_apply_inverse
    nb = quat_apply_inverse(env.scene["robot"].data.root_link_quat_w, terrain_normal(env, sensor_name))
    return (nb[:, :2] ** 2).sum(-1)


def foot_heights_now(env, sensor_name: str = "foot_height") -> torch.Tensor:
    """Each foot's sphere centre above the terrain under it: the site's current height minus the highest terrain hit
    of its ray ring (the rays are from the previous ``sense()``; the terrain under a foot barely moves in 20 ms); 0 when
    no ray hit."""
    d = env.scene[sensor_name].data
    B, F = d.heights.shape[0], d.heights.shape[1]
    hit_z = d.hit_pos_w[..., 2].view(B, F, -1)
    hit = (d.distances.view(B, F, -1) >= 0)
    ground = torch.where(hit, hit_z, torch.full_like(hit_z, -1.0e3)).amax(-1)
    site = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 2]
    # no hit: the rays started inside the terrain (a foot pressed in past its centre); read 0, as the sensor does
    return torch.where(hit.any(-1), site - ground, torch.zeros_like(site))


def foot_heights_centre(env, sensor_name: str = "foot_height") -> torch.Tensor:
    """As ``foot_heights_now`` but from each ring's centre ray only (straight under the foot): the foot's height
    without the ring's uphill bias on a slope (for measuring how far a foot is pressed in)."""
    d = env.scene[sensor_name].data
    B, F = d.heights.shape[0], d.heights.shape[1]
    hit_z = d.hit_pos_w[..., 2].view(B, F, -1)[..., 0]
    hit = d.distances.view(B, F, -1)[..., 0] >= 0
    site = env.scene["robot"].data.site_pos_w[:, _foot_site_ids(env), 2]
    return torch.where(hit, site - hit_z, torch.zeros_like(site))


def foot_clearance_now(env, target_height: float, command_name: str = "twist", command_threshold: float = 0.02):
    """mjlab's ``feet_clearance`` (|height - target| weighted by the foot's horizontal speed, while commanded) on
    ``foot_heights_now``."""
    v = env.scene["robot"].data.site_lin_vel_w[:, _foot_site_ids(env), :2].norm(dim=-1)
    cost = ((foot_heights_now(env) - target_height).abs() * v).sum(-1)
    cmd = env.command_manager.get_command(command_name)
    active = (cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()) > command_threshold
    return cost * active.float()


def abduction_deviation_l2(env):
    """The four abductions' squared distance from the stand (measured angles, rad): nothing else keeps the stance
    while walking."""
    return (mdp.servo_action(env).true_rel()[:, list(ABDUCTION)] ** 2).sum(-1)


def leg_contacts(env, sensor_name: str = "leg_contact"):
    return (env.scene[sensor_name].data.found > 0).float().sum(-1)


def obs_body_scan(env, sensor_name: str = "body_scan", clip: float = 0.15):
    """Heights of the base above the terrain at the scan points, minus the stand height, clipped (critic only)."""
    h = envs_mdp.height_scan(env, sensor_name)
    return (h - 0.0725).clamp(-clip, clip)


def obs_foot_heights(env, sensor_name: str = "foot_height"):
    return (env.scene[sensor_name].data.heights - FOOT_R).clamp(-0.05, 0.10)


def heading_align(env, command_name: str = "twist", std: float = 0.25):
    """exp(-(error / std)^2) of the true heading error to the loop's target (now: rewards run before the command
    update), on the envs the loop drives; 0 elsewhere."""
    from mjlab.utils.lab_api.math import wrap_to_pi
    t = env.command_manager.get_term(command_name)
    e = wrap_to_pi(t.heading_target - t.robot.data.heading_w)
    return torch.exp(-(e / std) ** 2) * (t.is_heading_env & ~t.is_standing_env).float()


# ---------------------------------------------------------------------------------------------------- the task
def built_flights(opt: TerrainOptions) -> dict:
    """"stairs_built": one straight flight per riser in ``stair_risers`` (``stair_steps`` risers, ``stair_tread``
    apart), keyed "built" (one riser) or "built_<mm>mm"."""
    out = {}
    for r in opt.stair_risers:
        tag = "built" if len(opt.stair_risers) == 1 else f"built_{1000 * r:g}mm"
        layout = ((float(r), float(opt.stair_tread)),) * int(opt.stair_steps)
        out[tag] = terrain_mod.StairGridCfg(size=(opt.tile, opt.tile), layout=layout, width=opt.stair_width,
                                            landing=opt.stair_landing, tag=tag)
    return out


def sub_terrains(opt: TerrainOptions) -> dict:
    if opt.terrain == "stairs_narrow":
        return terrain_mod.stairs_narrow_sub_terrains(opt.tile, riser_range=opt.narrow_risers)
    if opt.terrain in TERRAIN_PRESETS:
        return TERRAIN_PRESETS[opt.terrain](opt.tile)
    if opt.terrain == "stairs_built":  # straight flights of the given risers (a replica of a built staircase)
        return built_flights(opt)
    if opt.terrain == "flat":
        from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
        return {"flat": BoxFlatTerrainCfg(size=(opt.tile, opt.tile))}
    raise ValueError(f"unknown terrain {opt.terrain!r}")


def make_terrain_env_cfg(opt: TerrainOptions | None = None):
    opt = opt or TerrainOptions()
    cfg = make_env_cfg(opt)                     # the forward task, then changed below
    arm = arms_mod.get(opt.arm)
    subs = sub_terrains(opt)
    rows = 1 if opt.terrain == "stairs_built" else opt.rows
    gen = TerrainGeneratorCfg(size=(opt.tile, opt.tile), sub_terrains=subs, num_rows=rows, curriculum=True,
                              border_width=BORDER, seed=opt.terrain_seed)
    cfg.scene.terrain = TerrainEntityCfg(terrain_type="generator", terrain_generator=gen,
                                         max_init_terrain_level=opt.max_init_level if opt.level is None else 0)
    cfg.scene.entities = {"robot": robot_cfg(arm.actuators, strip_visuals=not opt.visuals,
                                             masses=opt.masses, self_collision=opt.self_collision,
                                             foot_solref=opt.foot_solref)}
    half_x = rows * opt.tile / 2 + BORDER
    half_y = len(subs) * opt.tile / 2 + BORDER

    # sensors: the forward ones, a scan around the body, each foot's height over the terrain, leg-terrain contacts
    body_scan = RayCastSensorCfg(name="body_scan", frame=ObjRef(type="body", name="base", entity="robot"),
                                 ray_alignment="yaw", pattern=GridPatternCfg(size=(0.30, 0.18), resolution=0.03),
                                 max_distance=1.0, exclude_parent_body=True, include_geom_groups=(0,))
    foot_height = TerrainHeightSensorCfg(name="foot_height",
                                         frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
                                         pattern=RingPatternCfg.single_ring(radius=0.006, num_samples=4),
                                         ray_alignment="yaw", max_distance=0.5, exclude_parent_body=True,
                                         include_geom_groups=(0,), reduction="min")
    leg_contact = ContactSensorCfg(name="leg_contact",                     # thigh and shin capsules, not the feet
                                   primary=ContactMatch(mode="geom", pattern=LEG_COLLISION_GEOMS, entity="robot"),
                                   secondary=ContactMatch(mode="body", pattern="terrain"),
                                   fields=("found",), reduce="none", num_slots=1)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (body_scan, foot_height, leg_contact)
    if opt.clearance == "apex":
        cfg.scene.sensors += (RayCastSensorCfg(name="foot_ahead", frame=foot_height.frame,
                                               pattern=AheadPatternCfg(size=(0.03, 0.01), resolution=0.01),
                                               ray_alignment="yaw", max_distance=0.3, exclude_parent_body=True,
                                               include_geom_groups=(0,)),)

    # observations: the omni command form (vx, vy, wz); the critic sees the terrain
    for g in ("actor", "anchor", "critic"):
        cfg.observations[g].terms["command"] = ObservationTermCfg(func=mdp.obs_command, params={"omni": True})
    cfg.observations["critic"].terms["body_scan"] = ObservationTermCfg(func=obs_body_scan)
    cfg.observations["critic"].terms["foot_heights"] = ObservationTermCfg(func=obs_foot_heights)
    if opt.cross_track:      # the drift the cross-track term charges, seen by the critic (appended last: widen_critic)
        cfg.observations["critic"].terms["track_errors"] = ObservationTermCfg(func=mdp.obs_track_errors)
    if opt.history:
        cfg.observations["actor"].history_length = opt.history

    # commands
    fixed = opt.fixed_command
    assert opt.heading in ("integral", "loop"), opt.heading
    loop = opt.heading == "loop" and not fixed
    narrow_cols = tuple(i for i, c in enumerate(subs.values()) if isinstance(c, terrain_mod.NarrowFlightCfg))
    assert opt.heading_law in ("p", "pi"), opt.heading_law
    assert not narrow_cols or (loop and opt.stair_exit), "narrow flights need --heading loop and --stair-exit"
    assert loop or fixed or not opt.heading_reward, "the heading reward needs --heading loop"
    cfg.commands = {"twist": RelativeHeadingCommandCfg(
        heading_law=opt.heading_law, heading_wz=opt.heading_wz, narrow_cols=narrow_cols if loop else (),
        entity_name="robot", resampling_time_range=(1.0e6, 1.0e6) if fixed else tuple(opt.resample_s),
        heading_command=loop, heading_control_stiffness=opt.heading_kp,
        rel_standing_envs=0.0 if fixed else opt.rel_standing, rel_heading_envs=opt.rel_heading if loop else 0.0,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(fixed[0], fixed[0]) if fixed else (0.0, opt.vx_max), lin_vel_y=(0.0, 0.0),
            ang_vel_z=(fixed[2], fixed[2]) if fixed else (-opt.wz_max, opt.wz_max),
            heading=(-opt.heading_turn, opt.heading_turn) if loop else None))}

    # events
    if narrow_cols and not fixed:   # narrow flights: face the flight, as the robot is placed at the foot of the stairs
        cfg.events["reset_base"] = EventTermCfg(func=reset_base_narrow, mode="reset", params={
            "pose_range": {"yaw": (-math.pi, math.pi)}, "narrow_cols": narrow_cols,
            "narrow_pose_range": {"x": (-0.03, 0.03), "y": (-0.01, 0.01), "yaw": (-math.radians(2), math.radians(2))}})
    else:
        cfg.events["reset_base"] = EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                                params={"pose_range": {"yaw": (-math.pi, math.pi)} if not fixed else {},
                                                        "velocity_range": {}})
    if opt.randomise:
        cfg.events["foot_friction"].params["ranges"] = tuple(opt.friction_range)
        cfg.events["base_mass"] = EventTermCfg(
            func=envs_mdp.dr.body_mass, mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "operation": "scale",
                    "ranges": (0.9, 1.1)})
        if opt.foot_softness is not None:
            cfg.events["foot_softness"] = EventTermCfg(
                func=foot_softness, mode="startup",
                params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                        "ranges": tuple(opt.foot_softness)})
        if opt.push_mps > 0:
            p = opt.push_mps
            cfg.events["push"] = EventTermCfg(func=envs_mdp.push_by_setting_velocity, mode="interval",
                                              interval_range_s=(3.0, 6.0),
                                              params={"velocity_range": {"x": (-p, p), "y": (-p, p)}})

    if opt.yaw_catch:     # catches: yaw torque pulses on the base (not tied to randomise: a disturbance, not a robot property)
        cfg.events["yaw_catch"] = EventTermCfg(func=YawCatch, mode="step", params={
            "torque": tuple(opt.yaw_catch), "duration_s": (0.2, 0.4), "cooldown_s": (3.0, 6.0),
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)), "start_step": opt.yaw_catch_start})

    # rewards
    r = cfg.rewards
    for k in ("flat_orientation", "yaw", "lateral_position", "lateral_velocity", "swing_progress"):
        r.pop(k, None)
    r["track_ang_vel"] = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=1.0, params={"std": 0.25})
    if opt.heading == "integral":
        r["heading"] = RewardTermCfg(func=mdp.HeadingTrack, weight=-2.0, params={})
    if opt.alive:
        r["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=opt.alive)
    if opt.terminated:
        r["terminated"] = RewardTermCfg(func=envs_mdp.is_terminated, weight=opt.terminated)
    r["upright"] = RewardTermCfg(func=upright_terrain_l2, weight=-1.0)
    assert opt.clearance in ("speed", "apex"), opt.clearance
    if opt.clearance == "speed":
        r["foot_clearance"] = RewardTermCfg(func=foot_clearance_now, weight=-5.0,
                                            params={"target_height": 0.025 + FOOT_R})
    else:
        r["swing_apex"] = RewardTermCfg(func=SwingApexClearance, weight=opt.clearance_weight,
                                        params={"target": opt.clearance_target})
    if opt.progress:
        r["progress"] = RewardTermCfg(func=progress_linear, weight=opt.progress)
    if opt.stumble:
        r["stumble"] = RewardTermCfg(func=feet_stumble, weight=opt.stumble)
    r["air_time"] = RewardTermCfg(func=vel_mdp.feet_air_time, weight=0.25,
                                  params={"sensor_name": "feet_contact", "threshold_min": 0.05,
                                          "threshold_max": 0.30, "command_name": "twist",
                                          "command_threshold": 0.02})
    r["leg_contact"] = RewardTermCfg(func=leg_contacts, weight=opt.leg_contact)
    if opt.abduction:
        r["abduction"] = RewardTermCfg(func=abduction_deviation_l2, weight=opt.abduction)
    r["stand_still"] = RewardTermCfg(func=mdp.stand_still_l2, weight=-1.0, params={})
    r["path"] = RewardTermCfg(func=PathTracker, weight=1.0e-9, params={})
    if opt.cross_track:
        r["cross_track"] = RewardTermCfg(func=CrossTrackClipped, weight=opt.cross_track,
                                         params={"clip": opt.cross_track_clip})
    if opt.heading_reward:
        r["heading_align"] = RewardTermCfg(func=heading_align, weight=opt.heading_reward)

    # terminations and curriculum
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(func=out_of_bounds, time_out=True,
                                                           params={"half_x": half_x, "half_y": half_y})
    cfg.terminations["tilt"].params["limit_rad"] = opt.tilt_limit
    if opt.stair_exit:
        cfg.terminations["stair_exit"] = TerminationTermCfg(func=stair_exit, time_out=True,
                                                            params={"subs": tuple(subs.values()), "tile": opt.tile})
        if any(isinstance(c, terrain_mod.NarrowFlightCfg) for c in subs.values()):
            cfg.terminations["side_exit"] = TerminationTermCfg(func=side_exit, params={"subs": tuple(subs.values())})
    if opt.level is None and rows > 1:
        cfg.curriculum["terrain_levels"] = CurriculumTermCfg(func=terrain_levels_path,
                                                             params={"promote_m": opt.tile / 2})
        if len(subs) > 1:
            cfg.curriculum["family_levels"] = CurriculumTermCfg(func=family_levels, params={"names": tuple(subs)})
    cfg.sim.njmax = 600
    cfg.sim.nconmax = 64
    cfg.episode_length_s = opt.episode_s
    return cfg
