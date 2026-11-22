"""The scene on the GPU: an environment built on ``scene_cfg`` runs, with the servo models and the plant's draws; the
robot holding its stand settles at the kinematic stand height."""
import mujoco
import pytest
import torch

pytestmark = pytest.mark.gpu

N = 8


def _env(opt_kw, seconds=0.0):
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.envs import mdp as envs_mdp
    from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
    from mjlab.sim import MujocoCfg, SimulationCfg

    from mp4_sim.actuators import servo_model
    from mp4_sim.envs import scene as S

    sc = S.scene_cfg(S.SceneOptions(num_envs=N, **opt_kw))
    c = sc.legs.contract()
    knees = {f"{leg}_knee_drive": float(c.stand_rel_rad[3 * i + 2]) for i, leg in enumerate(("FR", "FL", "RR", "RL"))}
    from mjlab.managers.event_manager import EventTermCfg
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    resets = {"reset_base": EventTermCfg(func=envs_mdp.reset_root_state_uniform, mode="reset",
                                         params={"pose_range": {}, "velocity_range": {}}),
              "reset_joints": EventTermCfg(func=envs_mdp.reset_joints_by_offset, mode="reset",
                                           params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                                   "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))})}
    obs = {"actor": ObservationGroupCfg(terms={"q": ObservationTermCfg(func=envs_mdp.joint_pos_rel)},
                                        concatenate_terms=True, enable_corruption=False)}
    cfg = ManagerBasedRlEnvCfg(
        scene=sc.scene, observations=obs,
        actions={"joints": envs_mdp.JointPositionActionCfg(entity_name="robot", actuator_names=(".*_hip", ".*_thigh"),
                                                           scale=0.0, use_default_offset=True),
                 "knees": envs_mdp.TendonLengthActionCfg(entity_name="robot", actuator_names=(".*_knee_drive",),
                                                         scale=0.0, offset=knees)},
        events=S.ordered_events(sc, first=resets), rewards={}, terminations={}, curriculum={},
        sim=SimulationCfg(njmax=600, nconmax=64, mujoco=MujocoCfg(timestep=servo_model.PHYSICS_DT, iterations=10,
                                                                   ls_iterations=20)),
        decimation=20, episode_length_s=20.0)
    cfg.seed = 1
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0"), sc


@pytest.mark.parametrize("robot", ["sku2", "sku1"])
def test_env_on_the_scene_settles_at_the_stand(robot):
    from mp4_sim.robot import model
    env, sc = _env(dict(robot=robot, terrain="flat", rows=1, max_init_level=0, randomise=False, masses_json=None,
                        self_collision=True, foot_solref=(0.01, 1.0)))
    try:
        env.reset()
        acts = env.scene["robot"].actuators
        assert [type(a).__name__ for a in acts] == ["BamServo"] * 3
        z0 = float((env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).mean())
        assert z0 == pytest.approx(model.init_base_z(sc.legs), abs=1e-6)
        a = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")
        for _ in range(100):                         # 2 s
            env.step(a)
        p = env.scene["robot"].data.root_link_pos_w
        assert torch.isfinite(p).all()
        z = float((p[:, 2] - env.scene.env_origins[:, 2]).mean())
        assert z == pytest.approx(model.stand_geometry(sc.legs)["base_z_m"], abs=0.003)
        m = env.sim.mj_model
        t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "robot/FR_knee_drive")
        assert t >= 0
    finally:
        env.close()


def test_plant_draws_and_the_lidar():
    from scene_cases import CASES
    kw = {k: v for k, v in CASES["sku1_getup_com"].items() if k != "num_envs"}
    env, sc = _env({**kw, "supply_v_range": (10.5, 12.6)})
    try:
        env.reset()
        assert env.attachment_draw["mass_kg"].shape == (N,)
        assert env.com_height_draw.shape == (N,) and float(env.com_height_draw.std()) > 0
        fr = env.sim.model.geom_friction
        assert fr.shape[0] == N
        vin = env.scene["robot"].actuators[0].vin_tensor
        assert float(vin.min()) >= 10.5 and float(vin.max()) <= 12.6
        a = torch.zeros(N, env.action_manager.total_action_dim, device="cuda:0")
        for _ in range(10):
            env.step(a)
        assert torch.isfinite(env.scene["robot"].data.root_link_pos_w).all()
    finally:
        env.close()
