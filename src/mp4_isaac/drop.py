"""Drop the imported SKU2 onto a floor in Newton and check that it stays a robot.

The joints are held at the CAD assembly (every angle 0) by a plain implicit PD.
The CAD link inertias are about 1e-6 kg m^2, so the hold uses a small armature
and a small stiffness. A stiffness of a few N m/rad with no armature makes the
joint oscillator unstable and the state goes non-finite before the feet land.
This is a load test of the USD, not a gait. Run from the Isaac virtualenv:

    python -m mp4_isaac.drop
    python -m mp4_isaac.drop --video
"""

from __future__ import annotations

import argparse
import os
import sys

# Newton's GL viewer opens a window through pyglet. pyglet chooses that window
# class at import time, so a video run has to request the headless EGL path
# before Isaac Lab imports the viewer.
if any(arg == "--video" or arg.startswith("--video=") for arg in sys.argv[1:]):
    os.environ.pop("DISPLAY", None)
    os.environ.pop("WAYLAND_DISPLAY", None)
    import pyglet

    pyglet.options["headless"] = True

from isaaclab.app import add_launcher_args, launch_simulation

from .paths import CACHE, interface_usd, require_sku

parser = argparse.ArgumentParser(description="Drop SKU2 in Newton and check the articulation.")
parser.add_argument("--sku", default="sku_2")
parser.add_argument("--seconds", type=float, default=1.0, help="How long to simulate after reset.")
parser.add_argument(
    "--video",
    nargs="?",
    const=str(CACHE / "drop.mp4"),
    default=None,
    help="Write an mp4 of the drop. With no path, writes .cache/drop.mp4. No window.",
)
parser.add_argument("--stiffness", type=float, default=0.2, help="Implicit PD stiffness, N m/rad.")
parser.add_argument("--damping", type=float, default=0.02, help="Implicit PD damping, N m s/rad.")
parser.add_argument("--armature", type=float, default=1.0e-4, help="Joint armature, kg m^2.")
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

# The converted mesh bounds put the lowest point 82 mm under the base origin.
SPAWN_Z = 0.20
JOINTS = (
    "front_left_hip_joint",
    "front_left_leg_upper_joint",
    "front_left_leg_lower_joint",
    "front_right_hip_joint",
    "front_right_leg_upper_joint",
    "front_right_leg_lower_joint",
    "back_left_hip_joint",
    "back_left_leg_upper_joint",
    "back_left_leg_lower_joint",
    "back_right_hip_joint",
    "back_right_leg_upper_joint",
    "back_right_leg_lower_joint",
)


def usd_path() -> str:
    sku = require_sku(args_cli.sku)
    path = interface_usd(sku)
    if path is None:
        raise SystemExit(f"no USD for {sku} yet; run python -m mp4_isaac.convert --sku {sku}")
    return str(path)


@configclass
class DropSceneCfg(InteractiveSceneCfg):
    """One SKU2 over a static floor."""

    floor = AssetBaseCfg(
        prim_path="/World/Floor",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 2.0, 0.10),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path(), make_uninstanceable=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_Z), joint_pos={".*": 0.0}, joint_vel={".*": 0.0}),
        actuators={
            "joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=args_cli.stiffness,
                damping=args_cli.damping,
                armature=args_cli.armature,
                joint_effort_limit=3.0,
            )
        },
    )


def _tensor(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _report(robot, label: str) -> None:
    names = list(robot.joint_names)
    bodies = list(robot.body_names)
    pos = _tensor(robot.data.root_pos_w)[0].detach().cpu()
    quat = _tensor(robot.data.root_quat_w)[0].detach().cpu()
    q = _tensor(robot.data.joint_pos)[0].detach().cpu()
    qd = _tensor(robot.data.joint_vel)[0].detach().cpu()
    lowest = _tensor(robot.data.body_pos_w)[0, :, 2].detach().cpu().min()
    print(f"[{label}]")
    print(f"  bodies ({len(bodies)}): {', '.join(bodies)}")
    print(f"  joints ({len(names)}): {', '.join(names)}")
    print(f"  root pos m: {pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}")
    print(f"  root quat xyzw: {quat[0]:+.4f} {quat[1]:+.4f} {quat[2]:+.4f} {quat[3]:+.4f}")
    print(f"  lowest body z m: {float(lowest):+.4f}")
    print(f"  |q| max rad: {float(q.abs().max()):.4f}  |qd| max rad/s: {float(qd.abs().max()):.4f}")


def _failures(robot) -> list[str]:
    names = list(robot.joint_names)
    bodies = list(robot.body_names)
    pos = _tensor(robot.data.root_pos_w)[0]
    q = _tensor(robot.data.joint_pos)[0]
    qd = _tensor(robot.data.joint_vel)[0]
    lowest = _tensor(robot.data.body_pos_w)[0, :, 2].min()
    problems = []
    expected = set(JOINTS)
    if require_sku(args_cli.sku) == "sku_1":
        # The SKU1 SolidWorks export spells this joint back_riht. The MuJoCo builder renames it.
        expected.remove("back_right_leg_upper_joint")
        expected.add("back_riht_leg_upper_joint")
    if set(names) != expected:
        problems.append(f"joints {names} are not the twelve CAD joints")
    if not any(body.endswith("base_link") or body == "base_link" for body in bodies):
        problems.append("base_link is not a body")
    if any("display_bezel" == body or body.endswith("/display_bezel") for body in bodies):
        problems.append("the screen bezel is a rigid body")
    if any("display_panel" == body or body.endswith("/display_panel") for body in bodies):
        problems.append("the screen panel is a rigid body")
    finite = torch.isfinite(pos).all() and torch.isfinite(q).all() and torch.isfinite(qd).all()
    if not bool(finite):
        problems.append("state is not finite")
    z = float(pos[2])
    if z < 0.02 or z > 1.0:
        problems.append(f"root height {z:.3f} m is outside 0.02-1.0 m")
    if float(lowest) < -0.02:
        problems.append(f"a body is {float(lowest):.3f} m under the floor")
    if float(qd.abs().max()) > 30.0:
        problems.append(f"joint speed {float(qd.abs().max()):.1f} rad/s")
    return problems


def _viewer_cfg():
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

    return NewtonGLVisualizerCfg(
        headless=True,
        streaming_view=False,
        enable_picking=False,
        window_width=1280,
        window_height=720,
        eye=(0.32, -0.38, 0.18),
        lookat=(0.02, 0.0, 0.06),
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
    """Read the viewer image with glReadPixels.

    Newton's own readback maps the pixel buffer through CUDA/OpenGL interop.
    That interop is missing in this WSL session, and the copy fallback returns
    the same sky-and-floor frame every time.
    """
    import numpy as np
    from pyglet import gl

    fbo = viewer.renderer._frame_fbo
    fbo_id = _gl_name(fbo)
    width, height = viewer.renderer._screen_width, viewer.renderer._screen_height
    raw = (gl.GLubyte * (width * height * 3))()
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo_id)
    gl.glReadPixels(0, 0, width, height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, raw)
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, 0)
    image = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)[::-1]
    return np.ascontiguousarray(image)


def _rgb(sim) -> "np.ndarray":
    if not sim.visualizers:
        raise RuntimeError("the headless viewer was not created")
    viz = sim.visualizers[0]
    viz.render_rgb_array()
    return _framebuffer(viz._viewer)


def _write_video(path: str, frames: list, fps: int) -> None:
    import imageio.v2 as imageio

    output = os.path.abspath(path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"video: {output}")


def main() -> None:
    if args_cli.video:
        # The default --viz none would discard the headless viewer config.
        args_cli.visualizer = None
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        if args_cli.video:
            # The model is built before the viewer exists, so the default leaves
            # visual meshes out and the clip shows only collision hulls.
            physics_cfg.load_visual_shapes = True
        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device, physics=physics_cfg)
        if args_cli.video:
            sim_cfg.visualizer_cfgs = [_viewer_cfg()]
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view(eye=(0.32, -0.38, 0.18), target=(0.02, 0.0, 0.06))
        scene = instantiate(DropSceneCfg(num_envs=1, env_spacing=1.0))
        robot = scene["robot"]
        sim.reset()
        scene.update(sim.get_physics_dt())
        _report(robot, "reset")
        dt = sim.get_physics_dt()
        steps = max(1, int(round(args_cli.seconds / dt)))
        target = _tensor(robot.data.default_joint_pos).clone()
        every = max(1, int(round(0.1 / dt)))
        video_fps = 30
        video_stride = max(1, int(round((1.0 / video_fps) / dt)))
        frames = []
        for step in range(1, steps + 1):
            robot.actuators.target_command.set_position_index(value=target)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if args_cli.video and step % video_stride == 0:
                frames.append(_rgb(sim))
            q = _tensor(robot.data.joint_pos)[0]
            if step % every == 0 or not bool(torch.isfinite(q).all()):
                z = float(_tensor(robot.data.root_pos_w)[0, 2])
                print(
                    f"  t={step * dt:.3f}s  root z={z:+.4f}  "
                    f"|q|={float(torch.nan_to_num(q).abs().max()):.4f}"
                )
            if not bool(torch.isfinite(q).all()):
                break
        _report(robot, f"after {args_cli.seconds:.2f}s")
        if args_cli.video and frames:
            _write_video(args_cli.video, frames, video_fps)
        problems = _failures(robot)
        if problems:
            for problem in problems:
                print(f"FAIL {problem}")
            raise SystemExit(1)
        print("ok")


if __name__ == "__main__":
    main()
