"""The SKU2 policy contract as Isaac Lab MDP terms: the servo action, the robot-side observations, noise, rewards.

The action and the observations are ``contract`` in torch:

- action: clip to [-1, 1], ``scale * a`` about the stand, the band, the 20 degree step limit against the previous
  applied target, firmware tick quantisation. The target then waits one sensing-to-command chain delay (8-19 ms,
  drawn per episode) before it reaches the servos, which add their own fitted command delay.
- position and velocity: sampled every 5 ms as the AT32 reports them (an integer deci-degree), the velocity through
  ``contract.VelocityFilter`` (median of 3, then a 4-sample slope).
- gyro and gravity from an IMU at the base origin, gravity as ``-acc / |acc|``.

Servo coordinates map onto the CAD hinges through ``servo_map.json`` (``mapping.ServoMap``), the mapping
``play.py`` runs the shipped policies with. It is affine, so it is one 12x12 matrix here.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg, ManagerTermBase
from isaaclab.utils import configclass

from .contract import BAND_DEG, DD0, STATE_DT, STEP_LIMIT_DEG, Contract
from .mapping import LEG_ORDER, ServoMap
from .paths import asset_dir, contract_json, servo_map_json

URDF_PREFIX = {"FR": "front_right", "FL": "front_left", "RR": "back_right", "RL": "back_left"}
URDF_JOINTS = tuple(
    f"{URDF_PREFIX[leg]}_{suffix}" for leg in LEG_ORDER for suffix in ("hip_joint", "leg_upper_joint", "leg_lower_joint")
)
FEET = ".*_leg_lower_link"
THIGHS = ".*_leg_upper_link"
BASE = "base_link"
SAMPLE_EVERY = round(STATE_DT / 0.001)        # physics steps per STATE frame at 1 ms physics
CHAIN_MS = (8, 19)
F32 = torch.float32


def _t(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def stand_joint_pos() -> dict[str, float]:
    """The firmware stand (every servo at 0) on the CAD hinges."""
    q = ServoMap(servo_map_json()).stand_joint_angles()
    return {name: float(v) for name, v in zip(URDF_JOINTS, q)}


class ServoFrame:
    """Policy coordinates (rel, servo order, rad) <-> CAD hinge angles, and the firmware's units, in torch."""

    def __init__(self, device):
        c = Contract.load(contract_json())
        a, b = ServoMap(servo_map_json()).affine()                    # q = a @ servo_deg + b
        to_deg = np.degrees(1.0) * np.asarray(c.signs, dtype=float)   # servo_deg = degrees(rel) * signs
        m = a * to_deg[None, :]
        self.m_t = torch.tensor(m.T, dtype=F32, device=device)        # q = rel @ m_t + b
        self.m_inv_t = torch.tensor(np.linalg.inv(m).T, dtype=F32, device=device)
        self.b = torch.tensor(b, dtype=F32, device=device)
        self.signs = torch.tensor(c.signs, dtype=F32, device=device)
        self.offsets = torch.tensor(c.offsets_deg, dtype=F32, device=device)

    def joints(self, rel: torch.Tensor) -> torch.Tensor:
        return rel @ self.m_t + self.b

    def rel(self, q: torch.Tensor) -> torch.Tensor:
        return (q - self.b) @ self.m_inv_t

    def quantise(self, rel: torch.Tensor) -> torch.Tensor:
        """``Contract.quantise_rel``: the target the firmware sends, back in policy coordinates."""
        sdeg = torch.rad2deg(rel) * self.signs
        ticks = (511 + torch.trunc((sdeg + self.offsets) * 10240.0 / 3100.0)).clamp(0, 1023)
        dd = torch.trunc(3100.0 - ticks * 3100.0 / 1024.0)
        return torch.deg2rad((DD0 - dd) / 10.0 - self.offsets) * self.signs

    def measure(self, rel: torch.Tensor) -> torch.Tensor:
        """A true position as the AT32 reports it: an integer deci-degree on the servo's track."""
        sdeg = torch.rad2deg(rel) * self.signs
        dd = torch.round(DD0 - 10.0 * (sdeg + self.offsets))
        return torch.deg2rad((DD0 - dd) / 10.0 - self.offsets) * self.signs


def velocity_from_hist(h: torch.Tensor) -> torch.Tensor:
    """``contract.VelocityFilter``: a median of 3, then the 4-sample least-squares slope (h: (n, 6, 12))."""
    med = torch.stack([h[:, k:k + 3].median(dim=1).values for k in range(4)], dim=1)
    w = torch.tensor([-1.5, -0.5, 0.5, 1.5], device=h.device, dtype=h.dtype) / (5.0 * STATE_DT)
    return (w[None, :, None] * med).sum(1)


class ServoAction(ActionTerm):
    """Policy action (12, servo order) -> hinge position targets for ``PupperBamActuator``."""

    cfg: ServoActionCfg

    def __init__(self, cfg: ServoActionCfg, env):
        super().__init__(cfg, env)
        dev = self.device
        n = self.num_envs
        ids, names = self._asset.find_joints(list(URDF_JOINTS), preserve_order=True)
        if tuple(names) != URDF_JOINTS:
            raise RuntimeError(f"joint order {names} != {URDF_JOINTS}")
        self._jid = torch.tensor(ids, dtype=torch.long, device=dev)
        self.frame = ServoFrame(dev)
        self.scale = torch.tensor(cfg.scale, dtype=F32, device=dev).expand(12).clone()
        self.band = torch.deg2rad(torch.tensor(BAND_DEG, dtype=F32, device=dev))
        self.limit = math.radians(cfg.step_limit_deg)
        self._raw = torch.zeros(n, 12, device=dev)
        self.prev_rel = torch.zeros(n, 12, device=dev)
        self.applied_action = torch.zeros(n, 12, device=dev)
        self.target_q = self.frame.b.expand(n, 12).clone()
        self._queue_len = int(cfg.chain_ms[1]) + 1
        self._queue = self.frame.b.expand(n, self._queue_len, 12).clone()
        self._head = 0
        self.chain = torch.full((n,), int(cfg.chain_ms[0]), dtype=torch.long, device=dev)
        self._rows = torch.arange(n, device=dev)
        self.hist = torch.zeros(n, 6, 12, device=dev)
        self._substep = 0
        self._decimation = int(env.cfg.decimation)
        self._synced_at = -1
        self.reset(None)

    @property
    def action_dim(self) -> int:
        return 12

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self.target_q

    def true_rel(self) -> torch.Tensor:
        return self.frame.rel(_t(self._asset.data.joint_pos)[:, self._jid])

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        a = actions.clamp(-1.0, 1.0)
        rel = torch.maximum(torch.minimum(self.scale * a, self.band), -self.band)
        rel = torch.maximum(torch.minimum(rel, self.prev_rel + self.limit), self.prev_rel - self.limit)
        if self.cfg.quantise:
            rel = self.frame.quantise(rel)
        self.prev_rel = rel
        self.applied_action = rel / self.scale
        self.target_q = self.frame.joints(rel)

    def apply_actions(self) -> None:
        # Substep i sees the state after i physics steps. The 200 Hz samples are the states at 5, 10 and 15 ms
        # here and at the observation's own instant in ``sync``.
        i = self._substep % self._decimation
        if i and i % SAMPLE_EVERY == 0:
            self._push(self.frame.measure(self.true_rel()))
        self._substep += 1
        self._head = (self._head + 1) % self._queue_len
        self._queue[:, self._head] = self.target_q
        delayed = self._queue[self._rows, (self._head - self.chain) % self._queue_len]
        self._asset.set_joint_position_target_index(target=delayed, joint_ids=self._jid)

    def _push(self, x: torch.Tensor, env_ids=None) -> None:
        if env_ids is None:
            self.hist = torch.cat([self.hist[:, 1:], x[:, None, :]], dim=1)
        else:
            self.hist[env_ids] = x[env_ids, None, :].expand(-1, 6, -1)

    def reset(self, env_ids=None) -> None:
        ids = self._rows if env_ids is None else torch.as_tensor(env_ids, device=self.device).long()
        self._raw[ids] = 0.0
        self.prev_rel[ids] = 0.0
        self.applied_action[ids] = 0.0
        self.target_q[ids] = self.frame.b
        self._queue[ids] = self.frame.b
        lo, hi = self.cfg.chain_ms
        self.chain[ids] = torch.randint(int(lo), int(hi) + 1, (len(ids),), device=self.device)
        self._push(self.frame.measure(self.true_rel()), ids)

    def sync(self) -> None:
        """Push the state at the observation's instant, once per policy step."""
        step = int(self._env.common_step_counter)
        if step != self._synced_at:
            self._synced_at = step
            self._push(self.frame.measure(self.true_rel()))

    @property
    def pos_rel(self) -> torch.Tensor:
        return self.hist[:, -1]

    @property
    def velocity(self) -> torch.Tensor:
        return velocity_from_hist(self.hist)


@configclass
class ServoActionCfg(ActionTermCfg):
    class_type: type = ServoAction
    asset_name: str = "robot"
    scale: tuple = (0.25, 0.50, 0.50) * 4
    """rad per unit action, servo order (abduction, hip, knee) x 4."""
    step_limit_deg: float = STEP_LIMIT_DEG
    quantise: bool = True
    chain_ms: tuple = CHAIN_MS
    """Sensing-to-command delay drawn per episode, physics steps (ms) at 1 ms physics."""


def servo_action(env) -> ServoAction:
    return env.action_manager.get_term("servo")


# ------------------------------------------------------------------------------------------------ observation noise
NOISY = ("ang_vel", "gravity", "pos", "vel")
NOISE_START = {"ang_vel": 0.02, "gravity": 0.01, "pos": 0.002, "vel": 0.05}


def noise_profile(name: str) -> dict:
    """The measured end amplitudes: ``n1`` one half-width per channel, ``n2`` one per element."""
    path = asset_dir("sku_2") / f"noise_{name}.json"
    prof = json.loads(path.read_text())["profile_uniform_half_width"]
    return {k: (list(map(float, v)) if isinstance(v, list) else float(v)) for k, v in prof.items()}


class NoiseTable:
    """Uniform noise half-widths per observation channel; ``ramp_noise`` writes ``amp``."""

    def __init__(self, device, start: dict, end: dict):
        self.start = {k: torch.as_tensor(start[k], dtype=F32, device=device) for k in NOISY}
        self.end = {k: torch.as_tensor(end[k], dtype=F32, device=device) for k in NOISY}
        self.amp = dict(self.start)

    def apply(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return x + (torch.rand_like(x) * 2.0 - 1.0) * self.amp[name]


def install_noise(env, env_ids, start: dict, end: dict, ramp: bool):
    if not hasattr(env, "sim_noise"):
        env.sim_noise = NoiseTable(env.device, start, end)
        if not ramp:
            env.sim_noise.amp = dict(env.sim_noise.end)


def _maybe_noise(env, name, x, noisy):
    table = getattr(env, "sim_noise", None)
    return table.apply(name, x) if (noisy and table is not None) else x


def ramp_noise(env, env_ids, hold_iters: int, ramp_iters: int, steps_per_iter: int = 24):
    """Hold the start amplitudes, then move linearly to the end amplitudes."""
    table = getattr(env, "sim_noise", None)
    if table is None:
        return 0.0
    it = env.common_step_counter // steps_per_iter
    frac = 0.0 if it <= hold_iters else min(1.0, (it - hold_iters) / max(ramp_iters, 1))
    for k in NOISY:
        table.amp[k] = table.start[k] + frac * (table.end[k] - table.start[k])
    return frac


# ------------------------------------------------------------------------------------------------- observations
def obs_ang_vel(env, noisy: bool = True):
    return _maybe_noise(env, "ang_vel", _t(env.scene["imu"].data.ang_vel_b), noisy)


def obs_gravity(env, noisy: bool = True):
    acc = _t(env.scene["imu"].data.lin_acc_b)
    g = -acc / acc.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return _maybe_noise(env, "gravity", g, noisy)


def obs_command(env, command_name: str = "twist", omni: bool = True):
    cmd = env.command_manager.get_command(command_name)
    if omni:
        return cmd.clone()
    out = torch.zeros_like(cmd)
    out[:, 0] = cmd[:, 0]
    return out


def obs_pos(env, noisy: bool = True):
    term = servo_action(env)
    term.sync()
    return _maybe_noise(env, "pos", term.pos_rel, noisy)


def obs_vel(env, noisy: bool = True):
    term = servo_action(env)
    term.sync()
    return _maybe_noise(env, "vel", term.velocity, noisy)


def obs_prev_action(env):
    return servo_action(env).applied_action


def obs_true_lin_vel(env):
    return _t(env.scene["robot"].data.root_lin_vel_b)


def obs_base_height(env):
    return _t(env.scene["robot"].data.root_pos_w)[:, 2:3] - env.scene.env_origins[:, 2:3]


def obs_foot_contact(env, sensor_name: str = "contact_forces", threshold: float = 1.0):
    sensor = env.scene.sensors[sensor_name]
    ids = _feet(env, sensor_name)
    return (_t(sensor.data.net_forces_w)[:, ids].norm(dim=-1) > threshold).float()


def _feet(env, sensor_name: str):
    key = f"_mp4_feet_{sensor_name}"
    if not hasattr(env, key):
        ids, _ = env.scene.sensors[sensor_name].find_sensors(FEET, preserve_order=True)
        setattr(env, key, torch.as_tensor(ids, dtype=torch.long, device=env.device))
    return getattr(env, key)


# ------------------------------------------------------------------------------------------------------ events
def body_masses(env, env_ids, heatsink: bool, base_scale: tuple, com_xy: float, com_z: float):
    """The weighed masses (``masses.apply_masses``), then per env a base mass scale and a base COM offset."""
    from .masses import apply_masses

    robot = env.scene["robot"]
    apply_masses(robot, "sku_2", heatsink=heatsink)
    names = list(robot.body_names)
    base = names.index(BASE)
    n = env.num_envs
    masses = _t(robot.data.body_mass).detach().clone()
    inertia = _t(robot.data.body_inertia).detach().clone()
    k = torch.empty(n, device=masses.device).uniform_(*base_scale)
    masses[:, base] *= k
    inertia[:, base] *= k[:, None]
    com_pos = _t(robot.data.body_com_pos_b).detach().clone()
    com_quat = _t(robot.data.body_com_quat_b).detach().clone()
    com_pos[:, base, 0:2] += torch.empty(n, 2, device=masses.device).uniform_(-com_xy, com_xy)
    com_pos[:, base, 2] += torch.empty(n, device=masses.device).uniform_(-com_z, com_z)
    robot.set_inertias_index(inertias=inertia)
    robot.set_masses_index(masses=masses)
    robot.set_coms_index(coms=torch.cat([com_pos, com_quat], dim=-1))


# ------------------------------------------------------------------------------------------------------ rewards
def track_lin_vel_xy_exp(env, std: float, command_name: str = "twist"):
    v = _t(env.scene["robot"].data.root_lin_vel_b)[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    return torch.exp(-((v - cmd) ** 2).sum(-1) / std**2)


def track_ang_vel_z_exp(env, std: float, command_name: str = "twist"):
    w = _t(env.scene["robot"].data.root_ang_vel_b)[:, 2]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    return torch.exp(-((w - cmd) ** 2) / std**2)


def servo_torques_l2(env):
    """The servo model's motor torque in servo coordinates (the knee on its tendon)."""
    act = env.scene["robot"].actuators["servos"]
    return (act.servo_motor_torque**2).sum(-1)


def stand_still_l2(env, command_name: str = "twist", deadband: float = 1e-3):
    """With no command, the joints' squared distance from the stand."""
    cmd = env.command_manager.get_command(command_name)
    idle = (cmd.abs().amax(dim=-1) < deadband).float()
    return idle * (servo_action(env).true_rel() ** 2).sum(-1)


def feet_air_time(env, threshold_min: float, threshold_max: float, command_name: str = "twist",
                  sensor_name: str = "contact_forces", command_threshold: float = 0.02):
    """At touchdown, the air time just ended, clipped to [min, max] and counted from min; while commanded."""
    sensor = env.scene.sensors[sensor_name]
    ids = _feet(env, sensor_name)
    first = _t(sensor.compute_first_contact(env.step_dt))[:, ids].float()
    air = _t(sensor.data.last_air_time)[:, ids]
    gain = (air.clamp(max=threshold_max) - threshold_min) * first
    cmd = env.command_manager.get_command(command_name)
    moving = (cmd[:, :2].norm(dim=-1) + cmd[:, 2].abs()) > command_threshold
    return gain.sum(-1) * moving.float()


def thigh_contacts(env, sensor_name: str = "contact_forces", threshold: float = 1.0):
    key = f"_mp4_thighs_{sensor_name}"
    sensor = env.scene.sensors[sensor_name]
    if not hasattr(env, key):
        ids, _ = sensor.find_sensors(THIGHS, preserve_order=True)
        setattr(env, key, torch.as_tensor(ids, dtype=torch.long, device=env.device))
    f = _t(sensor.data.net_forces_w)[:, getattr(env, key)]
    return (f.norm(dim=-1) > threshold).float().sum(-1)


def abduction_l2(env):
    """The four abductions' squared distance from the stand."""
    return (servo_action(env).true_rel()[:, 0::3] ** 2).sum(-1)


class Progress(ManagerTermBase):
    """Forward progress along the commanded direction per second, normalised by the commanded speed (1 = walking at
    the command), while the command asks for travel."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._prev = torch.zeros(env.num_envs, 2, device=env.device)
        self._fresh = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self._fresh[:] = True
        else:
            self._fresh[env_ids] = True

    def __call__(self, env, command_name: str = "twist", min_speed: float = 0.02):
        d = env.scene["robot"].data
        p = _t(d.root_pos_w)[:, :2]
        prev = torch.where(self._fresh[:, None], p, self._prev)
        self._prev = p.clone()
        self._fresh[:] = False
        cmd = env.command_manager.get_command(command_name)[:, :2]
        q = _t(d.root_quat_w)
        x, y, z, w = q.unbind(-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = torch.cos(yaw), torch.sin(yaw)
        v_w = torch.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], dim=-1)
        speed = v_w.norm(dim=-1)
        along = ((p - prev) * v_w).sum(-1) / speed.clamp_min(1e-6) / env.step_dt
        return (along / speed.clamp_min(1e-6)).clamp(-1.0, 1.0) * (speed > min_speed).float()


# -------------------------------------------------------------------------------------------------- terminations
def tilt_beyond(env, limit_rad: float):
    g = _t(env.scene["robot"].data.projected_gravity_b)
    return torch.acos((-g[:, 2]).clamp(-1.0, 1.0)) > limit_rad


def base_contact(env, sensor_name: str = "contact_forces", threshold: float = 1.0):
    key = f"_mp4_base_{sensor_name}"
    sensor = env.scene.sensors[sensor_name]
    if not hasattr(env, key):
        ids, _ = sensor.find_sensors(BASE)
        setattr(env, key, torch.as_tensor(ids, dtype=torch.long, device=env.device))
    f = _t(sensor.data.net_forces_w)[:, getattr(env, key)]
    return (f.norm(dim=-1) > threshold).any(dim=-1)
