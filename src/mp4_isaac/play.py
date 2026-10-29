"""Run an mp4-sim ONNX policy on the Isaac SKU2.

The policy's joints are firmware servo angles. ``mp4_sim.robot.mapping.ServoMap``
turns those into the CAD hinge angles this articulation uses, and back. The knee
is still the URDF calf hinge: the map already folds the servo's shin angle into
that hinge. Run from the Isaac virtualenv:

    python -m mp4_isaac.play --video
"""

from __future__ import annotations

import argparse
import json
import os
import sys

if any(arg == "--video" or arg.startswith("--video=") for arg in sys.argv[1:]):
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    import pyglet

    pyglet.options["headless"] = True

from isaaclab.app import add_launcher_args, launch_simulation

import numpy as np

from mp4_sim.contract import ActionDecoder, Contract, HistoryStacker, VelocityFilter, assemble_observation
from mp4_sim.robot.kinematics import LEG_ORDER
from mp4_sim.robot.mapping import ServoMap

from .paths import CACHE, interface_usd

POLICY = CACHE.parents[1] / "mp4-sim" / "policies" / "flat_omni" / "policy.onnx"
# CACHE is mp4-isaac/.cache, parents[1] is Haris. POLICY path:
# CACHE = repo/.cache, repo = mp4-isaac, parent = Haris. So parents[1] is Haris. Good.

URDF_PREFIX = {"FR": "front_right", "FL": "front_left", "RR": "back_right", "RL": "back_left"}
URDF_PART = (("hip", "hip_joint"), ("thigh", "leg_upper_joint"), ("calf", "leg_lower_joint"))
URDF_JOINTS = tuple(
    f"{URDF_PREFIX[leg]}_{suffix}" for leg in LEG_ORDER for _part, suffix in URDF_PART
)

parser = argparse.ArgumentParser(description="Run flat_omni on the Isaac SKU2.")
parser.add_argument("--policy", type=str, default=str(POLICY))
parser.add_argument("--seconds", type=float, default=4.0, help="Time the policy runs, after the stand settles.")
parser.add_argument("--settle", type=float, default=0.4, help="Hold the stand before the policy starts.")
parser.add_argument("--vx", type=float, default=0.15, help="Forward speed command, m/s.")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0)
parser.add_argument("--stiffness", type=float, default=1.5)
parser.add_argument("--damping", type=float, default=0.08)
parser.add_argument("--armature", type=float, default=5.0e-4)
parser.add_argument(
    "--video",
    nargs="?",
    const=str(CACHE / "flat_omni.mp4"),
    default=None,
    help="Write an mp4. With no path, writes .cache/flat_omni.mp4.",
)
parser.add_argument("--physics", default="newton_mjwarp", choices=["newton_mjwarp"])
add_launcher_args(parser)
parser.set_defaults(visualizer=["none"])
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass, instantiate

SPAWN_Z = 0.12
MAP = ServoMap()
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
            size=(6.0, 6.0, 0.10),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path(), make_uninstanceable=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_Z), joint_pos=STAND, joint_vel={".*": 0.0}),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=args_cli.stiffness,
                damping=args_cli.damping,
                armature=args_cli.armature,
                joint_effort_limit=1.2,
            )
        },
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


def _read_model_joints(robot, index: list[int]) -> tuple[np.ndarray, np.ndarray]:
    q = _tensor(robot.data.joint_pos)[0].detach().cpu().numpy()
    qd = _tensor(robot.data.joint_vel)[0].detach().cpu().numpy()
    return q[index], qd[index]


def _targets(robot, index: list[int], model_q: np.ndarray) -> torch.Tensor:
    current = _tensor(robot.data.joint_pos)
    target = torch.zeros_like(current)
    target[0, index] = torch.as_tensor(model_q, dtype=target.dtype, device=target.device)
    return target


def _rel(model_q: np.ndarray, contract: Contract) -> np.ndarray:
    return contract.rel_from_servo_deg(MAP.joint_to_servo(model_q))


def _imu(robot) -> tuple[np.ndarray, np.ndarray]:
    ang = _tensor(robot.data.root_ang_vel_b)[0].detach().cpu().numpy()
    quat = _tensor(robot.data.root_quat_w)[0].detach().cpu().numpy()
    gravity = _quat_xyzw_to_matrix(quat).T @ np.array([0.0, 0.0, -1.0])
    return ang, gravity


def _root(robot) -> np.ndarray:
    return _tensor(robot.data.root_pos_w)[0].detach().cpu().numpy()


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
        eye=(0.45, -0.55, 0.28),
        lookat=(0.15, 0.0, 0.06),
    )


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


def main() -> None:
    session, obs_name, manifest = load_policy(args_cli.policy)
    if int(manifest.get("history", 1)) != 1 or not manifest["omni"]:
        raise SystemExit(f"this runner expects flat_omni (history 1, omni); got {manifest.get('history')} {manifest['omni']}")
    contract = Contract.load()
    decoder = ActionDecoder(contract, manifest["action_scale_rad"])
    history = HistoryStacker(1)
    velocity = VelocityFilter()
    command = np.array([args_cli.vx, args_cli.vy, args_cli.wz], dtype=float)

    if args_cli.video:
        args_cli.visualizer = None
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if args_cli.video:
            physics_cfg.load_visual_shapes = True
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        if args_cli.video:
            sim_cfg.visualizer_cfgs = [_viewer_cfg()]
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(0.45, -0.55, 0.28), target=(0.15, 0.0, 0.06))
        scene = instantiate(PlaySceneCfg(num_envs=1, env_spacing=1.0))
        robot = scene["robot"]
        sim.reset()
        scene.update(sim.get_physics_dt())
        index = _indices(robot)
        dt = sim.get_physics_dt()
        control_every = max(1, int(round(float(manifest["control_dt"]) / dt)))
        settle_steps = int(round(args_cli.settle / dt))
        policy_steps = int(round(args_cli.seconds / dt))
        stand = _targets(robot, index, STAND_Q)
        frames = []
        video_stride = max(1, int(round((1.0 / 30.0) / dt)))
        start_xy = _root(robot)[:2].copy()
        action = np.zeros(12)

        for step in range(1, settle_steps + policy_steps + 1):
            q, _qd = _read_model_joints(robot, index)
            rel = _rel(q, contract)
            velocity.push(rel)
            if step == settle_steps + 1:
                decoder.reset()
                history.reset()
            if step <= settle_steps:
                target = stand
            elif (step - settle_steps - 1) % control_every == 0:
                ang, gravity = _imu(robot)
                obs = assemble_observation(ang, gravity, command, rel, velocity.velocity(), decoder.prev_action)
                action = session.run(None, {obs_name: history(obs)[None, :]})[0][0]
                applied = decoder(action)
                target = _targets(robot, index, MAP.servo_to_joint(contract.servo_deg_from_rel(applied["rel"])))
            robot.actuators.target_command.set_position_index(value=target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if args_cli.video and step % video_stride == 0:
                frames.append(_rgb(sim))
            if not np.isfinite(q).all():
                print("FAIL state is not finite")
                raise SystemExit(1)

        pos = _root(robot)
        ang, gravity = _imu(robot)
        q, _qd = _read_model_joints(robot, index)
        travelled = float(np.linalg.norm(pos[:2] - start_xy))
        print(f"command: vx={args_cli.vx:.2f} vy={args_cli.vy:.2f} wz={args_cli.wz:.2f}")
        print(f"root: x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}  travelled {travelled:.3f} m")
        print(f"gravity body: {gravity[0]:+.3f} {gravity[1]:+.3f} {gravity[2]:+.3f}")
        print(f"|servo rel| max {float(np.max(np.abs(_rel(q, contract)))):.3f} rad")
        if args_cli.video and frames:
            _write_video(args_cli.video, frames, 30)
        if pos[2] < 0.03:
            print("FAIL the base is on the floor")
            raise SystemExit(1)
        print("ok")


if __name__ == "__main__":
    main()
