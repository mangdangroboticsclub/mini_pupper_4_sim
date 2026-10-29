"""Run a shipped ONNX policy on the Isaac SKU2.

``flat_omni`` walks the ground plane. ``rough_terrain`` and ``stairs`` were
trained with a heading loop on the terrain in ``terrain.py``, and this runner
builds that terrain and closes the loop. Physics is 1 ms. The velocity filter
still sees a sample every 5 ms. Run from the Isaac virtualenv:

    python -m mp4_isaac.play --policy rough_terrain --video
    python -m mp4_isaac.play --policy stairs --video
    python -m mp4_isaac.play --policy stairs --crowd 5 --video
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

if any(arg == "--video" or arg.startswith("--video=") for arg in sys.argv[1:]):
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    import pyglet

    pyglet.options["headless"] = True

from isaaclab.app import add_launcher_args, launch_simulation

import numpy as np

from mp4_sim.contract import (
    STATE_DT,
    ActionDecoder,
    Contract,
    HistoryStacker,
    VelocityFilter,
    assemble_observation,
)
from mp4_sim.heading import pi_step
from mp4_sim.robot.kinematics import LEG_ORDER
from mp4_sim.robot.mapping import ServoMap

from .masses import apply_masses
from .paths import CACHE, PHYSICS_DT, contract_json, interface_usd, policies, policy_path, servo_map_json
from .servos import PupperBamActuatorCfg
from .terrain import assets as terrain_assets
from .terrain import build as build_course
from .terrain import crowd_layout, stair_formation

URDF_PREFIX = {"FR": "front_right", "FL": "front_left", "RR": "back_right", "RL": "back_left"}
URDF_PART = (("hip", "hip_joint"), ("thigh", "leg_upper_joint"), ("calf", "leg_lower_joint"))
URDF_JOINTS = tuple(
    f"{URDF_PREFIX[leg]}_{suffix}" for leg in LEG_ORDER for _part, suffix in URDF_PART
)

parser = argparse.ArgumentParser(description="Run a shipped policy on the Isaac SKU2.")
parser.add_argument("--policy", type=str, default="flat_omni", help=f"Policy name or an .onnx path. Shipped: {', '.join(policies())}.")
parser.add_argument("--terrain", default=None, choices=["flat", "rough", "stairs"], help="Default follows the policy.")
parser.add_argument("--crowd", type=int, default=1, help="How many robots run the policy, side by side.")
parser.add_argument("--seconds", type=float, default=4.0, help="Time the policy runs, after the stand settles.")
parser.add_argument("--settle", type=float, default=0.4, help="Hold the stand before the policy starts.")
parser.add_argument("--vx", type=float, default=None, help="Forward speed command, m/s. Terrain policies default to 0.12.")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument(
    "--heatsink",
    action="store_true",
    help="Use the 478.66 g SKU2 budget (Raspberry Pi heatsink). Default is 450 g.",
)
parser.add_argument(
    "--video",
    nargs="?",
    const="auto",
    default=None,
    help="Write an mp4. With no path, writes .cache/<policy>.mp4.",
)
parser.add_argument("--physics", default="newton_mjwarp", choices=["newton_mjwarp"])
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args_cli = parser.parse_args()

POLICY_NAME = Path(args_cli.policy).stem if args_cli.policy.endswith(".onnx") else args_cli.policy
POLICY_FILE = policy_path(args_cli.policy)
TERRAIN_FOR = {"rough_terrain": "rough", "stairs": "stairs"}
if args_cli.terrain is None:
    args_cli.terrain = TERRAIN_FOR.get(POLICY_NAME, "flat")
if args_cli.vx is None:
    args_cli.vx = 0.12 if args_cli.terrain != "flat" else 0.15
if args_cli.terrain != "flat":
    args_cli.heatsink = True
if args_cli.crowd < 1:
    raise SystemExit("--crowd must be at least 1")
if args_cli.video == "auto":
    suffix = "_crowd" if args_cli.crowd > 1 else ""
    args_cli.video = str(CACHE / f"{POLICY_NAME}{suffix}.mp4")
COURSE = build_course(args_cli.terrain, lanes=args_cli.crowd if args_cli.terrain == "stairs" else 1)
SPAWN = (COURSE.spawn[0], COURSE.spawn[1], COURSE.spawn[2] + 0.12)

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, instantiate

SPAWN_Z = 0.12
MAP = ServoMap(servo_map_json())
STAND_Q = MAP.stand_joint_angles()
STAND = {name: float(value) for name, value in zip(URDF_JOINTS, STAND_Q)}


def usd_path() -> str:
    path = interface_usd("sku_2")
    if path is None:
        raise SystemExit("no USD for sku_2 yet; run python -m mp4_isaac.convert --sku sku_2")
    return str(path)


@configclass
class PlaySceneCfg(InteractiveSceneCfg):
    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(12.0, 8.0, 0.10) if args_cli.terrain == "stairs" and args_cli.crowd > 1 else (
                (8.0, 4.0, 0.10) if args_cli.terrain != "flat" else (6.0, 6.0, 0.10)
            ),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.72, 0.80, 0.90) if args_cli.terrain == "stairs" and args_cli.crowd > 1 else (0.45, 0.45, 0.45)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.0, -0.05) if args_cli.terrain != "flat" else (0.0, 0.0, -0.05)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path(), make_uninstanceable=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=SPAWN, joint_pos=STAND, joint_vel={".*": 0.0}),
        actuators={"servos": PupperBamActuatorCfg(physics_dt=PHYSICS_DT)},
    )


def _tensor(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _indices(robot) -> list[int]:
    names = list(robot.joint_names)
    missing = [name for name in URDF_JOINTS if name not in names]
    if missing:
        raise RuntimeError(f"USD is missing joints: {missing}")
    return [names.index(name) for name in URDF_JOINTS]


def _read_model_joints(robot, index: list[int]) -> np.ndarray:
    q = _tensor(robot.data.joint_pos).detach().cpu().numpy()
    return q[:, index]


def _fill_targets(robot, index: list[int], model_q: np.ndarray) -> torch.Tensor:
    """``model_q`` is (robots, 12) in URDF joint order."""
    current = _tensor(robot.data.joint_pos)
    target = torch.zeros_like(current)
    target[:, index] = torch.as_tensor(model_q, dtype=target.dtype, device=target.device)
    return target


def _rel(model_q: np.ndarray, contract: Contract) -> np.ndarray:
    return contract.rel_from_servo_deg(MAP.joint_to_servo(model_q))


def _imu_row(ang: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gravity = _quat_xyzw_to_matrix(quat).T @ np.array([0.0, 0.0, -1.0])
    return ang, gravity


def _roots(robot) -> np.ndarray:
    return _tensor(robot.data.root_pos_w).detach().cpu().numpy()


def load_policy(path: str):
    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    meta = {item.key: item.value for item in model.metadata_props}
    manifest = json.loads(meta["mp3_manifest_json"])
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session, session.get_inputs()[0].name, manifest


def _viewer_cfg():
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    return NewtonGLVisualizerCfg(
        headless=True,
        streaming_view=False,
        enable_picking=False,
        window_width=1280,
        window_height=720,
        eye=COURSE.camera_eye,
        lookat=COURSE.camera_look,
    )


def _yaw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _gl_name(value) -> int:
    if isinstance(value, int):
        return value
    native = getattr(value, "value", None)
    if isinstance(native, int):
        return native
    if isinstance(native, (bytes, bytearray)):
        return int.from_bytes(native[:4], "little")
    return int.from_bytes(bytes(value)[:4], "little")


def _framebuffer(viewer):
    from pyglet import gl

    fbo = viewer.renderer._frame_fbo
    width, height = viewer.renderer._screen_width, viewer.renderer._screen_height
    raw = (gl.GLubyte * (width * height * 3))()
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, _gl_name(fbo))
    gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, raw)
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
    image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)[::-1]
    return np.ascontiguousarray(image)


def _rgb(sim):
    viz = sim.visualizers[0]
    viz.render_rgb_array()
    return _framebuffer(viz._viewer)


def _write_video(path: str, frames: list, fps: int) -> None:
    import imageio.v2 as imageio

    output = os.path.abspath(path)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"video: {output}")


class _Mind:
    """One robot's policy state. The network weights are shared; this is the memory."""

    def __init__(self, contract: Contract, scale, history_length: int):
        self.decoder = ActionDecoder(contract, scale)
        self.history = HistoryStacker(history_length)
        self.velocity = VelocityFilter()
        self.command = np.array([args_cli.vx, args_cli.vy, args_cli.wz], dtype=float)
        self.heading_i = 0.0
        self.yaw0 = 0.0


def main() -> None:
    session, obs_name, manifest = load_policy(str(POLICY_FILE))
    if not manifest["omni"]:
        raise SystemExit(f"{POLICY_NAME} is not an omni policy")
    contract = Contract.load(contract_json())
    history_length = int(manifest.get("history", 1))
    minds = [_Mind(contract, manifest["action_scale_rad"], history_length) for _ in range(args_cli.crowd)]
    heading = args_cli.terrain != "flat"
    print(
        f"policy: {POLICY_NAME}  terrain: {args_cli.terrain}  crowd: {args_cli.crowd}  "
        f"history: {history_length}  heading loop: {heading}"
    )

    if args_cli.video:
        args_cli.visualizer = None
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if args_cli.video:
            physics_cfg.load_visual_shapes = True
        # The BAM law is an explicit Lab actuator. Newton's native actuators only
        # run implicit PD and a few built-in explicit models.
        sim_cfg = sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            device=args_cli.device,
            physics=physics_cfg,
            use_newton_actuators=False,
        )
        if args_cli.video:
            sim_cfg.visualizer_cfgs = [_viewer_cfg()]
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=COURSE.camera_eye, target=COURSE.camera_look)
        scene_cfg = PlaySceneCfg(num_envs=args_cli.crowd, env_spacing=1.0)
        lanes = args_cli.crowd if args_cli.terrain == "stairs" else 1
        for name, asset in terrain_assets(args_cli.terrain, lanes=lanes).items():
            setattr(scene_cfg, name, asset)
        scene = instantiate(scene_cfg)
        robot = scene["robot"]
        sim.reset()
        scene.update(sim.get_physics_dt())
        apply_masses(robot, "sku_2", heatsink=args_cli.heatsink)
        index = _indices(robot)
        dt = sim.get_physics_dt()
        # Reset leaves the hinges at the CAD zero. The servo law is stiff enough
        # that a step from zero to the stand runs the joints away.
        n = args_cli.crowd
        posed = _tensor(robot.data.joint_pos).clone()
        posed[:, index] = torch.as_tensor(STAND_Q, dtype=posed.dtype, device=posed.device)
        robot.write_joint_position_to_sim_index(position=posed)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(posed))
        # The cloner parks the robots on its own grid. On the stair field each
        # rank stands on its own blue tread; otherwise they stand on the flat.
        if args_cli.terrain == "stairs" and n > 1:
            slots, _, _ = stair_formation(n)
        else:
            xy, _, _ = crowd_layout(n)
            slots = np.column_stack([xy[:, 0] + SPAWN[0], xy[:, 1] + SPAWN[1], np.full(n, SPAWN[2] - 0.12)])
        pose = torch.zeros(n, 7, dtype=posed.dtype, device=posed.device)
        pose[:, 0] = torch.as_tensor(slots[:, 0], dtype=posed.dtype, device=posed.device)
        pose[:, 1] = torch.as_tensor(slots[:, 1], dtype=posed.dtype, device=posed.device)
        pose[:, 2] = torch.as_tensor(slots[:, 2] + 0.12, dtype=posed.dtype, device=posed.device)
        pose[:, 6] = 1.0
        robot.write_root_link_pose_to_sim_index(root_pose=pose)
        robot.write_root_link_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, dtype=posed.dtype, device=posed.device))
        scene.update(dt)
        control_every = max(1, int(round(float(manifest["control_dt"]) / dt)))
        sample_every = max(1, int(round(STATE_DT / dt)))
        settle_steps = int(round(args_cli.settle / dt))
        policy_steps = int(round(args_cli.seconds / dt))
        stand = np.repeat(STAND_Q[None, :], n, axis=0)
        target = _fill_targets(robot, index, stand)
        frames = []
        video_stride = max(1, int(round((1.0 / 30.0) / dt)))
        start_xy = _roots(robot)[:, :2].copy()

        for step in range(1, settle_steps + policy_steps + 1):
            q = _read_model_joints(robot, index)
            quats = _tensor(robot.data.root_quat_w).detach().cpu().numpy()
            angs = _tensor(robot.data.root_ang_vel_b).detach().cpu().numpy()
            if (step - 1) % sample_every == 0:
                for i, mind in enumerate(minds):
                    mind.velocity.push(_rel(q[i], contract))
            if step == settle_steps + 1:
                for i, mind in enumerate(minds):
                    mind.decoder.reset()
                    mind.history.reset()
                    mind.yaw0 = _yaw(quats[i])
            if step <= settle_steps:
                target = _fill_targets(robot, index, stand)
            elif (step - settle_steps - 1) % control_every == 0:
                commanded = np.zeros((n, 12))
                for i, mind in enumerate(minds):
                    rel = _rel(q[i], contract)
                    ang, gravity = _imu_row(angs[i], quats[i])
                    if heading:
                        mind.command[2], mind.heading_i = pi_step(
                            _wrap(mind.yaw0 - _yaw(quats[i])), mind.heading_i, float(manifest["control_dt"])
                        )
                    obs = assemble_observation(ang, gravity, mind.command, rel, mind.velocity.velocity(), mind.decoder.prev_action)
                    action = session.run(None, {obs_name: mind.history(obs)[None, :]})[0][0]
                    commanded[i] = MAP.servo_to_joint(contract.servo_deg_from_rel(mind.decoder(action)["rel"]))
                target = _fill_targets(robot, index, commanded)
            robot.actuators.target_command.set_position_index(value=target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if args_cli.video and step % video_stride == 0:
                frames.append(_rgb(sim))
            if not np.isfinite(q).all():
                print(f"FAIL state is not finite at t={step * dt:.4f}s")
                raise SystemExit(1)

        pos = _roots(robot)
        travelled = np.linalg.norm(pos[:, :2] - start_xy, axis=1)
        print(f"command: vx={args_cli.vx:.2f}  crowd: {n}")
        for i in range(n):
            quat = _tensor(robot.data.root_quat_w)[i].detach().cpu().numpy()
            _, gravity = _imu_row(np.zeros(3), quat)
            print(
                f"  [{i}] x={pos[i, 0]:+.3f} y={pos[i, 1]:+.3f} z={pos[i, 2]:+.3f}  "
                f"travelled {travelled[i]:.3f} m  gravity z {gravity[2]:+.3f}"
            )
        if args_cli.video and frames:
            _write_video(args_cli.video, frames, 30)
        if float(pos[:, 2].min()) < 0.03:
            print("FAIL a base is on the floor")
            raise SystemExit(1)
        print("ok")


if __name__ == "__main__":
    main()
