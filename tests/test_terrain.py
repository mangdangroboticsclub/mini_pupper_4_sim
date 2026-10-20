"""Terrain: the generator's surfaces, the curriculum rule, the leg-base collisions, the warm start and resume paths;
on the GPU, the terrain task's observations, sensor timing, commands and terminations."""
import math
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mjlab")
import mujoco  # noqa: E402
import torch  # noqa: E402

from mp4_sim.gpu import terrain  # noqa: E402
from mp4_sim.gpu.terrain_task import terrain_levels_path  # noqa: E402
from mp4_sim.robot import sku2  # noqa: E402

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the GPU")
INIT_SCALE = np.array([0.32, 0.45, 0.42] * 4, dtype=np.float32)


@pytest.fixture(scope="module")
def init_policy(tmp_path_factory):
    """A 45-input omni policy (random weights) exported as ONNX: the warm start of the warm_start variant."""
    from mp4_sim.export import manifest, onnx
    rng = np.random.default_rng(7)
    w, b, prev = [], [], 45
    for n in (128, 128, 64, 12):
        w.append(rng.normal(0.0, 1.0 / np.sqrt(prev), (n, prev)).astype(np.float32))
        b.append(rng.normal(0.0, 0.05, n).astype(np.float32))
        prev = n
    out = tmp_path_factory.mktemp("init") / "stops"
    onnx.write(out, w, b, manifest.build(INIT_SCALE, omni=True))
    return out


def _train(*args):
    """``python -m mp4_sim.gpu.train`` on 4 envs of flat ground with the heading loop, checked."""
    cmd = [sys.executable, "-B", "-m", "mp4_sim.gpu.train", "--variant", "warm_start", "--envs", "4", "--terrain", "flat",
           "--heading", "loop", "--allow-unpatched", *map(str, args)]
    subprocess.run(cmd, check=True, capture_output=True)


def test_low_poly_periodic_edges_match():
    h = terrain.low_poly_heights(np.random.default_rng(0), (0.2, 0.2), 0.002, 0.02, 0.03, periodic=True)
    assert h.min() == 0.0 and 0.01 < h.max() <= 0.02 + 1e-9
    assert np.allclose(h[:, 0], h[:, -1], atol=1e-9) and np.allclose(h[0, :], h[-1, :], atol=1e-9)


def _curriculum_env(path, cmd, fell):
    calls = {}
    t = SimpleNamespace(terrain_levels=torch.zeros(len(path)),
                        update_env_origins=lambda ids, up, down: calls.update(up=up.clone(), down=down.clone()))
    env = SimpleNamespace(scene=SimpleNamespace(terrain=t), sim_path=torch.tensor(path),
                          sim_cmd_path=torch.tensor(cmd),
                          termination_manager=SimpleNamespace(terminated=torch.tensor(fell)))
    return env, calls


def test_curriculum_on_the_path():
    # walked far / fell / walked little / standing (too little commanded) / the first reset (nothing walked)
    env, calls = _curriculum_env([0.9, 0.9, 0.2, 0.0, 0.0], [1.2, 1.2, 1.2, 0.1, 0.0],
                                 [False, True, False, False, False])
    terrain_levels_path(env, torch.arange(5), promote_m=0.8)
    assert calls["up"].tolist() == [True, False, False, False, False]
    assert calls["down"].tolist() == [False, True, True, False, False]


def test_leg_base_collisions():
    spec = sku2.get_spec(strip_visuals=True, self_collision=True)
    base = [g for g in spec.body("base").geoms if g.contype or g.conaffinity]
    assert base and all(g.conaffinity & 4 for g in base)
    for leg in sku2.LEGS:
        assert all(g.contype & 4 for g in spec.body(f"{leg}_calf").geoms if g.contype or g.conaffinity)
        assert not any(g.contype & 4 for g in spec.body(f"{leg}_thigh").geoms)
    plain = sku2.get_spec(strip_visuals=True)
    assert not any(g.conaffinity & 4 for g in plain.body("base").geoms)


@needs_gpu
def test_terrain_task_observations_and_fresh_foot_heights():
    """The actor keeps the 45-float contract; the foot-height sensor is read after the step that moved the feet (a
    sensor one step stale would differ while the feet move)."""
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_sim.gpu.terrain_task import FOOT_R, TerrainOptions, make_terrain_env_cfg
    opt = TerrainOptions(arm="stock", num_envs=4, terrain="flat", noisy=False, randomise=False)
    env = ManagerBasedRlEnv(cfg=make_terrain_env_cfg(opt), device="cuda:0")
    obs, _ = env.reset()
    assert obs["actor"].shape == (4, 45) and obs["critic"].shape[1] > 45
    robot = env.scene["robot"]
    sites = [robot.site_names.index(s) for s in sku2.FOOT_SITES]
    g = torch.Generator(device="cuda:0").manual_seed(0)
    from mp4_sim.gpu.terrain_task import foot_heights_now
    worst, worst_now = 0.0, 0.0
    for _ in range(20):
        env.step(torch.rand(4, 12, device="cuda:0", generator=g) * 2 - 1)
        z = robot.data.site_pos_w[:, sites, 2]                 # flat terrain: its top is z = 0
        h = env.scene["foot_height"].data.heights              # what the observations read, after sense()
        now = foot_heights_now(env)                            # what the rewards read
        inside = z < 0                                         # a foot pressed in past its centre: rays start inside
        ref = torch.where(inside, torch.zeros_like(z), z)
        worst = max(worst, float((h - ref).abs().max()))
        worst_now = max(worst_now, float((now - ref).abs().max()))
    env.close()
    assert worst < 1e-4, worst
    assert worst_now < 1e-4, worst_now
    assert FOOT_R > 0


@needs_gpu
def test_history_stacker_is_mjlabs_history():
    """The robot side's numpy history (contract.HistoryStacker) rebuilds the actor's history observation from the plain
    45-float observations step by step, through a reset."""
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_sim.contract import HistoryStacker
    from mp4_sim.gpu.terrain_task import TerrainOptions, make_terrain_env_cfg
    opt = TerrainOptions(arm="stock", num_envs=2, terrain="flat", noisy=False, randomise=False, history=5)
    env = ManagerBasedRlEnv(cfg=make_terrain_env_cfg(opt), device="cuda:0")
    obs, _ = env.reset()
    stacks = [HistoryStacker(5), HistoryStacker(5)]
    g = torch.Generator(device="cuda:0").manual_seed(0)
    worst = 0.0
    for k in range(30):
        plain = obs["anchor"].cpu().numpy()                    # the same terms, clean, without history
        for i in range(2):
            worst = max(worst, float(np.abs(stacks[i](plain[i]) - obs["actor"][i].cpu().numpy()).max()))
        obs, _, term, trunc, _ = env.step(torch.rand(2, 12, device="cuda:0", generator=g) * 2 - 1)
        for i in np.flatnonzero((term | trunc).cpu().numpy()):
            stacks[i].reset()
    env.close()
    assert obs["actor"].shape[1] == 225
    assert worst < 1e-6, worst


@needs_gpu
def test_standing_robot_has_no_leg_contacts_progress_or_empty_currents():
    """Standing on its feet: no thigh or shin contact (the feet are not counted); commanded forward but held at the
    stand, the forward progress the curriculum reads stays near 0 (sway is not progress); the BAM currents fill all
    12 joint columns (knee tendons map to the calf columns)."""
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_sim.gpu import evaluate as te
    from mp4_sim.gpu.terrain_task import TerrainOptions, leg_contacts, make_terrain_env_cfg
    opt = TerrainOptions(arm="bam_t5ub06", num_envs=4, terrain="flat", noisy=False, randomise=False,
                         fixed_command=(0.15, 0.0, 0.0))
    env = ManagerBasedRlEnv(cfg=make_terrain_env_cfg(opt), device="cuda:0")
    env.reset()
    g = torch.Generator(device="cuda:0").manual_seed(0)
    seen = torch.zeros(12, dtype=torch.bool, device="cuda:0")
    for k in range(100):
        a = torch.zeros(4, 12, device="cuda:0") if k < 50 else (torch.rand(4, 12, device="cuda:0", generator=g) - 0.5) * 0.4
        env.step(a)
        if k == 49:
            assert float(leg_contacts(env).max()) == 0.0
            assert float(env.sim_path.max()) < 0.03            # 1 s commanded at 0.15 m/s, held at the stand
            assert float(env.sim_cmd_path.min()) > 0.1
        seen |= (te.bam_currents(env) > 1e-4).any(0)
    env.close()
    assert bool(seen.all()), seen.tolist()


def test_rescaled_warm_start_keeps_the_targets(tmp_path, init_policy):
    """policy_to_checkpoint: the same robot state (observation with prev_action in the new units) gives the same
    targets in radians as the original policy did with its own units."""
    from mp4_sim.export.onnx import load_policy
    from mp4_sim.gpu import train as tt
    new_scale = np.array([0.25, 0.5, 0.5] * 4, dtype=np.float32)
    ck = torch.load(tt.policy_to_checkpoint(init_policy, new_scale, tmp_path / "c.pt"),
                    weights_only=False)["actor_state_dict"]
    d = load_policy(init_policy)
    old_scale = d["scale"]

    def run(layers, x):
        for i, (w, b) in enumerate(layers):
            x = x @ w.T + b
            if i < len(layers) - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x
    old = list(zip(d["weights"], d["biases"]))
    new = [(ck[f"mlp.{2 * i}.weight"].numpy(), ck[f"mlp.{2 * i}.bias"].numpy()) for i in range(len(old))]
    obs = np.random.default_rng(0).normal(0, 0.5, (64, 45)).astype(np.float32)
    obs_new = obs.copy()
    obs_new[:, 33:45] *= old_scale / new_scale                 # prev_action = target / scale
    assert np.allclose(run(old, obs) * old_scale, run(new, obs_new) * new_scale, atol=1e-5)


@needs_gpu
def test_heading_loop_commands_and_reward_terms():
    """With heading="loop": the heading envs' wz is clip(kp * error) to a target drawn within ±heading_turn of the
    heading at the resample, the HeadingTrack penalty is gone, and the alive and termination terms are in."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import wrap_to_pi

    from mp4_sim.gpu.terrain_task import TerrainOptions, make_terrain_env_cfg
    opt = TerrainOptions(arm="stock", num_envs=64, terrain="flat", noisy=False, randomise=False, heading="loop",
                         rel_heading=1.0, rel_standing=0.0, alive=0.5, terminated=-50.0)
    cfg = make_terrain_env_cfg(opt)
    assert "heading" not in cfg.rewards and {"alive", "terminated"} <= set(cfg.rewards)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    env.step(torch.zeros(64, 12, device="cuda:0"))
    cmd = env.command_manager.get_term("twist")
    robot = env.scene["robot"]
    rel = wrap_to_pi(cmd.heading_target - robot.data.heading_w)
    assert float(rel.abs().max()) <= opt.heading_turn + 0.05      # one step of turning at most
    wz = env.command_manager.get_command("twist")[:, 2]
    want = torch.clip(opt.heading_kp * cmd.heading_error, -opt.wz_max, opt.wz_max)
    assert torch.allclose(wz, want, atol=1e-5)
    env.close()


def test_progress_linear_pays_progress_not_standing():
    from mp4_sim.gpu.terrain_task import progress_linear
    cmd = torch.tensor([[0.10, 0.0, 0.0], [0.10, 0.0, 0.0], [0.10, 0.0, 0.0], [0.01, 0.0, 0.0], [0.10, 0.0, 0.0]])
    v = torch.tensor([[0.0, 0.0], [0.05, 0.0], [0.20, 0.0], [0.01, 0.0], [0.0, 0.10]])  # stand/half/fast/tiny/sideways
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda name: cmd),
                          scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=v))})
    assert torch.allclose(progress_linear(env), torch.tensor([0.0, 0.5, 1.0, 0.0, 0.0]))


def test_feet_stumble_counts_sideways_contacts():
    from mp4_sim.gpu.terrain_task import feet_stumble
    force = torch.tensor([[[0.0, 0.0, 2.0], [3.0, 0.0, 1.0], [1.0, 0.0, 1.0], [5.0, 0.0, 0.1]]])
    found = torch.tensor([[1, 1, 1, 0]])                       # the last foot is in the air (no contact)
    env = SimpleNamespace(scene={"feet_contact": SimpleNamespace(data=SimpleNamespace(found=found, force=force))})
    assert float(feet_stumble(env)[0]) == 1.0                   # only the second: |F_xy| 3 > 1.5 * 1


def test_swing_apex_clearance_charges_a_low_swing_at_touchdown(monkeypatch):
    """A swing whose highest clearance over the terrain ahead stayed at 10 mm pays 15 mm (target 25) at touchdown; one
    that reached 30 mm pays nothing; nothing is paid mid-swing or while standing still."""
    from mp4_sim.gpu import terrain_task as tt
    n = 1
    monkeypatch.setattr(tt, "ground_ahead", lambda env: torch.zeros(n, 4))
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, _sim_foot_sites=[0, 1, 2, 3])
    env.scene = {"feet_contact": SimpleNamespace(data=SimpleNamespace(found=None)),
                 "robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=None))}
    env.command_manager = SimpleNamespace(get_command=lambda name: torch.tensor([[0.1, 0.0, 0.0]]))
    term = tt.SwingApexClearance(None, env)

    def step(found, apex_mm):
        env.scene["feet_contact"].data.found = torch.tensor([found], dtype=torch.int32)
        z = torch.full((n, 4), tt.FOOT_R)
        z[0, :] = torch.tensor([tt.FOOT_R + a / 1000.0 for a in apex_mm])
        env.scene["robot"].data.site_pos_w = torch.stack([torch.zeros_like(z), torch.zeros_like(z), z], -1)
        return float(term(env, target=0.025)[0]) * env.step_dt
    assert step([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0                         # all down
    assert step([0, 1, 0, 1], [5, 0, 5, 0]) == 0.0                         # FR, RR lift
    assert step([0, 1, 0, 1], [10, 0, 30, 0]) == 0.0                       # mid-swing: nothing
    assert step([0, 1, 0, 1], [8, 0, 20, 0]) == 0.0                        # 0.06 s in the air
    assert abs(step([1, 1, 1, 1], [0, 0, 0, 0]) - 0.015) < 1e-6            # touchdown: FR 25-10, RR 0 (30 > 25)
    assert step([1, 0, 1, 1], [0, 0, 0, 0]) == 0.0                         # FL flickers off for one step
    assert step([1, 1, 1, 1], [0, 0, 0, 0]) == 0.0                         # and back: too short to be a swing


def test_ahead_rays_start_high_and_ahead():
    from mp4_sim.gpu.terrain_task import AheadPatternCfg
    off, _ = AheadPatternCfg(size=(0.03, 0.01), resolution=0.01).generate_rays(None, "cpu")
    assert abs(float(off[:, 0].min())) < 1e-6 and abs(float(off[:, 0].max()) - 0.03) < 1e-6
    assert torch.allclose(off[:, 2], torch.full_like(off[:, 2], 0.05))


def test_raised_relief_pit_has_vertical_sides():
    from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
    cfg = terrain.RaisedReliefCfg(size=(1.6, 1.6))
    h, (r0, c0, ny, nx) = cfg.heights(0.5, np.random.default_rng(0))
    edge = 0.005 + 0.5 * 0.020
    assert np.all(h[r0:r0 + ny, c0:c0 + nx] == 0.0) and h[r0 - 1, c0:c0 + nx].min() >= edge - 1e-9
    gen = TerrainGenerator(TerrainGeneratorCfg(size=(1.6, 1.6), sub_terrains={"r": cfg}, num_rows=1, curriculum=True,
                                               border_width=0.5, seed=0))
    spec = mujoco.MjSpec()
    gen.compile(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    corner = gen._get_sub_terrain_position(0, 0)
    gid = np.zeros(1, np.int32)

    def z_at(x, i):
        return 1.0 - mujoco.mj_ray(m, d, np.array([corner[0] + x, corner[1] + i * cfg.pitch, 1.0]),
                                   np.array([0, 0, -1.0]), None, 1, -1, gid)
    rows = range(r0 + 5, r0 + ny - 5, 9)
    outside = [z_at(c0 * cfg.pitch - 0.0015, i) for i in rows]   # 1.5 mm outside the pit's -x side: the wall's top
    inside = [z_at(c0 * cfg.pitch + 0.0015, i) for i in rows]    # 1.5 mm inside: the pit floor
    assert min(outside) >= cfg.edge_range[0] - 1e-3 and max(inside) < 1e-4


@needs_gpu
def test_foot_softness_is_drawn_per_env_and_used():
    """Drawn per env within the range, the same on all four feet; a soft foot sinks deeper than a stiff one."""
    torch.manual_seed(0)                                        # the startup draws, whatever ran before
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_sim.gpu.terrain_task import TerrainOptions, foot_heights_centre, make_terrain_env_cfg
    opt = TerrainOptions(arm="stock", num_envs=16, terrain="flat", noisy=False, randomise=True, push_mps=0.0,
                         foot_softness=(0.005, 0.02), fixed_command=(0.0, 0.0, 0.0))
    env = ManagerBasedRlEnv(cfg=make_terrain_env_cfg(opt), device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    ids = [robot.indexing.geom_ids[robot.geom_names.index(g)].item() for g in sku2.FOOT_GEOMS]
    tc = env.sim.model.geom_solref[:, ids, 0]
    assert tc.shape[0] == 16 and float(tc.min()) >= 0.005 - 1e-6 and float(tc.max()) <= 0.02 + 1e-6
    assert float((tc - tc[:, :1]).abs().max()) < 1e-7 and float(tc[:, 0].std()) > 1e-3
    gs = env.sim.model.geom_solref
    gs[0, ids, 0] = 0.005                                       # in place (indexing with a list reads a copy)
    gs[1, ids, 0] = 0.02
    for _ in range(50):
        env.step(torch.zeros(16, 12, device="cuda:0"))
    from mp4_sim.gpu.terrain_task import FOOT_R
    pen = (FOOT_R - foot_heights_centre(env)).mean(-1)
    env.close()
    # standing: penetration scales about as timeconst^2 (16x from 0.005 to 0.02 s)
    assert float(pen[1]) > 5 * float(pen[0]) and float(pen[1]) > 1e-4, pen[:2].tolist()


@needs_gpu
def test_evaluation_seed_draws_the_robots():
    """evaluate.build seeds the env before its startup events: the same seed gives the same feet friction, another
    seed other values (seeding after construction would leave every seed on the same randomised robots)."""
    from mp4_sim.gpu import evaluate as te
    from mp4_sim.gpu.terrain_task import TerrainOptions
    opt = TerrainOptions(arm="stock", num_envs=8, terrain="flat", rows=1, noisy=False, randomise=True, level=0,
                         fixed_command=(0.1, 0.0, 0.0))
    fr = []
    for seed in (1, 1, 2):
        env = te.build(opt, seed=seed)
        robot = env.scene["robot"]
        ids = [robot.indexing.geom_ids[robot.geom_names.index(g)].item() for g in sku2.FOOT_GEOMS]
        fr.append(env.sim.model.geom_friction[:, ids[0], 0].clone().cpu())
        env.close()
    assert torch.equal(fr[0], fr[1]) and not torch.allclose(fr[0], fr[2])


@pytest.mark.parametrize("history", [5, 20])
def test_warm_start_with_history_acts_as_the_original(tmp_path, init_policy, history):
    """policy_to_checkpoint with history: on the stacked observation (HistoryStacker) the widened actor gives exactly
    the targets the 45-input warm start gives on the newest frame, whatever the older frames hold."""
    from mp4_sim.contract import HistoryStacker
    from mp4_sim.gpu import train as tt
    pol, scale = init_policy, np.array([0.25, 0.5, 0.5] * 4, dtype=np.float32)
    flat = torch.load(tt.policy_to_checkpoint(pol, scale, tmp_path / "a.pt"), weights_only=False)["actor_state_dict"]
    wide = torch.load(tt.policy_to_checkpoint(pol, scale, tmp_path / "b.pt", history=history),
                      weights_only=False)["actor_state_dict"]
    assert tuple(wide["mlp.0.weight"].shape) == (128, 45 * history)

    def run(sd, x):
        n = len([k for k in sd if k.endswith("weight")])
        for i in range(n):
            x = x @ sd[f"mlp.{2 * i}.weight"].numpy().T + sd[f"mlp.{2 * i}.bias"].numpy()
            if i < n - 1:
                x = np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
        return x
    rng = np.random.default_rng(0)
    st = HistoryStacker(history)
    for _ in range(history + 2):                                         # older frames differ from the newest
        o = rng.normal(0, 0.5, 45).astype(np.float32)
        stacked = st(o)
    assert np.allclose(run(wide, stacked[None]), run(flat, o[None]), atol=1e-5)


def test_abduction_deviation_charges_the_abductions_only(monkeypatch):
    from mp4_sim.gpu import terrain_task
    rel = torch.zeros(2, 12)
    rel[1, [0, 3, 6, 9]] = torch.tensor([0.1, -0.1, 0.2, 0.0])   # abductions off the stand
    rel[:, [1, 2, 4, 5]] = 0.5                                   # hips and knees do not count
    monkeypatch.setattr(terrain_task.mdp, "servo_action", lambda env: SimpleNamespace(true_rel=lambda: rel))
    assert torch.allclose(terrain_task.abduction_deviation_l2(None), torch.tensor([0.0, 0.06]))


@needs_gpu
def test_resume_restores_the_run(tmp_path, init_policy):
    """--resume: the run's history comes from its config and the actor, critic and iteration from the checkpoint
    (with no update, the saved checkpoint equals the source)."""
    _train("--init", init_policy, "--history", 2, "--iters", 1, "--out", tmp_path / "a")
    src = next((tmp_path / "a").glob("*/model_0.pt"))
    _train("--resume", src, "--iters", 0, "--abduction", -4, "--out", tmp_path / "b")
    run = next((tmp_path / "b").glob("*"))
    a, b = torch.load(src, weights_only=False), torch.load(run / "model_0.pt", weights_only=False)
    assert b["actor_state_dict"]["mlp.0.weight"].shape[1] == 90
    for k in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.equal(a[k][n], b[k][n]) for n in a[k])
    import json
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["variant_settings"]["history"] == 2 and cfg["resume"] == str(src) and cfg["lr0"] == 1e-4


def test_rough_steps_puts_the_stairs_at_60_percent():
    subs, b2 = terrain.rough_steps_sub_terrains(), terrain.rough_sub_terrains()
    assert abs(sum(c.proportion for c in subs.values()) - 1.0) < 1e-9
    assert abs(subs["stairs_up"].proportion - 0.30) < 1e-9 and abs(subs["stairs_down"].proportion - 0.30) < 1e-9
    assert subs["stairs_down"].step_height_range == b2["stairs_down"].step_height_range == (0.005, 0.030)
    assert abs(subs["low_poly"].proportion / subs["boxes"].proportion - 0.25 / 0.15) < 1e-9


def _stair_model(subs, rows, seed=0):
    from mjlab.terrains.terrain_generator import TerrainGenerator, TerrainGeneratorCfg
    terrain.STAIR_LAYOUTS.clear()
    gen = TerrainGenerator(TerrainGeneratorCfg(size=(1.6, 1.6), sub_terrains=subs, num_rows=rows, curriculum=True,
                                               border_width=0.5, seed=seed))
    spec = mujoco.MjSpec()
    gen.compile(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = np.zeros(1, np.int32)

    def z_at(row, col, x, y=0.8):
        c = gen._get_sub_terrain_position(row, col)
        return 1.0 - mujoco.mj_ray(m, d, np.array([c[0] + x, c[1] + y, 1.0]), np.array([0, 0, -1.0]), None, 1, -1, gid)
    return gen, z_at


@pytest.mark.parametrize("direction", ["climb", "descend"])
def test_stair_grid_risers_treads_and_vertical_faces(direction):
    """Each row's riser and each column's tread as specified; the faces vertical (a 1.5 mm step either side of a riser
    reads the two treads); the spawn on the approach, 0.15 m before the first riser, at the approach's height."""
    subs = terrain.stair_grid_sub_terrains(1.6, direction=direction, treads_mm=(50, 140))
    rows = len(terrain.STAIR_RISERS_MM)
    gen, z_at = _stair_model(subs, rows)
    for col, (tag, cfg) in enumerate(subs.items()):
        for row in (0, 6, rows - 1):
            h = terrain.STAIR_RISERS_MM[row] / 1000.0
            L = terrain.STAIR_LAYOUTS[(tag, row)]
            assert np.allclose(L["rises"], h) and np.allclose(L["treads"], cfg.tread) and len(L["rises"]) == 4
            xr = cfg.approach + np.arange(4) * cfg.tread
            sign = 1.0 if direction == "climb" else -1.0
            top0 = 0.0 if direction == "climb" else 4 * h
            for i, x in enumerate(xr):
                before, after = z_at(row, col, x - 0.0015), z_at(row, col, x + 0.0015)
                assert before == pytest.approx(top0 + sign * i * h, abs=1e-4)
                assert after == pytest.approx(top0 + sign * (i + 1) * h, abs=1e-4)
            o = gen.terrain_origins[row, col] - gen._get_sub_terrain_position(row, col)
            assert o == pytest.approx([cfg.approach - 0.15, 0.8, top0], abs=1e-9)
            assert L["risers_x"] == pytest.approx(list(xr - (cfg.approach - 0.15)))


def test_stair_grid_jitter_and_measured_layout():
    """Jitter: risers within ±30 % of the row's, treads within the clamp, reproducible by the terrain seed; a measured
    layout builds exactly those steps."""
    subs = terrain.stair_grid_sub_terrains(1.6, jitter=0.3, treads_mm=(50, 200))
    rows = len(terrain.STAIR_RISERS_MM)
    _stair_model(subs, rows, seed=3)
    first = {k: dict(v) for k, v in terrain.STAIR_LAYOUTS.items()}
    for (tag, row), L in first.items():
        h = terrain.STAIR_RISERS_MM[row] / 1000.0
        assert all(0.7 * h - 1e-12 <= r <= 1.3 * h + 1e-12 for r in L["rises"])
        assert all(0.05 - 1e-12 <= t <= 0.20 + 1e-12 for t in L["treads"])
        assert len(set(np.round(L["rises"], 9))) > 1
    _stair_model(subs, rows, seed=3)
    assert terrain.STAIR_LAYOUTS == first
    lay = ((0.021, 0.095), (0.018, 0.102), (0.024, 0.3))
    cfg = terrain.StairGridCfg(size=(1.6, 1.6), layout=lay, tag="built")
    _, z_at = _stair_model({"built": cfg}, 1)
    assert terrain.STAIR_LAYOUTS[("built", 0)]["rises"] == [0.021, 0.018, 0.024]
    assert z_at(0, 0, 0.45 + 0.095 + 0.102 + 0.01) == pytest.approx(0.063, abs=1e-4)


def test_stair_grid_width_and_landing():
    """A narrow flight: the treads only ``width`` wide, centred on the spawn line, the floor at 0 beside them; the top
    tread ends ``landing`` past the last riser (the floor beyond); recorded in the layout."""
    lay = ((0.025, 0.05), (0.025, 0.05), (0.025, 0.05), (0.025, 0.3))
    cfg = terrain.StairGridCfg(size=(1.6, 1.6), layout=lay, width=0.20, landing=0.25, tag="built")
    _, z_at = _stair_model({"built": cfg}, 1)
    x_top = 0.45 + 3 * 0.05
    assert z_at(0, 0, 0.45 + 0.01, y=0.8 + 0.095) == pytest.approx(0.025, abs=1e-4)    # inside the width
    assert z_at(0, 0, 0.45 + 0.01, y=0.8 + 0.105) == pytest.approx(0.0, abs=1e-4)      # beside it: the floor
    assert z_at(0, 0, x_top + 0.10, y=0.8 - 0.095) == pytest.approx(0.100, abs=1e-4)   # on the landing
    assert z_at(0, 0, x_top + 0.26) == pytest.approx(0.0, abs=1e-4)                     # past its end
    L = terrain.STAIR_LAYOUTS[("built", 0)]
    assert L["width"] == 0.20 and L["landing"] == 0.25


@needs_gpu
def test_resume_with_a_wider_action_scale_acts_as_before(tmp_path, init_policy):
    """--resume with --action-scale: the rescaled checkpoint gives the same joint targets (rad) and value on the same
    robot states (the previous action re-expressed in the new units), in the actor (with history) and the critic; the
    run's config carries the new scale."""
    import json
    _train("--init", init_policy, "--history", 2, "--iters", 1, "--out", tmp_path / "a")
    src = next((tmp_path / "a").glob("*/model_0.pt"))
    _train("--resume", src, "--iters", 0, "--action-scale", 0.25, 0.7, 0.7, "--out", tmp_path / "b")
    run = next((tmp_path / "b").glob("*"))
    a = torch.load(src, weights_only=False, map_location="cpu")
    b = torch.load(run / "resume_rescaled.pt", weights_only=False, map_location="cpu")
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["task"]["options"]["action_scale"] == [0.25, 0.7, 0.7] * 4
    old, new = np.array([0.25, 0.5, 0.5] * 4), np.array([0.25, 0.7, 0.7] * 4)
    k = torch.tensor(old / new, dtype=torch.float32)
    cols = b["rescale"]["prev_action_cols"]
    assert len(cols["actor"]) == 2 and len(cols["critic"]) == 1

    def mlp(sd, x):
        n = max(int(q.split(".")[1]) for q in sd if q.startswith("mlp."))
        for i in range(0, n + 1, 2):
            x = x @ sd[f"mlp.{i}.weight"].T + sd[f"mlp.{i}.bias"]
            if i < n:
                x = torch.nn.functional.elu(x)
        return x
    g = torch.Generator().manual_seed(0)
    for key, group in (("actor_state_dict", "actor"), ("critic_state_dict", "critic")):
        x = torch.randn(64, a[key]["mlp.0.weight"].shape[1], generator=g)
        xn = x.clone()
        for fc in cols[group]:
            xn[:, fc] = x[:, fc] * k            # the same previous target, in the new units
        ya, yb = mlp(a[key], x), mlp(b[key], xn)
        if group == "actor":
            ya, yb = ya * torch.tensor(old, dtype=torch.float32), yb * torch.tensor(new, dtype=torch.float32)
            assert torch.allclose(a[key]["distribution.std_param"] * torch.tensor(old, dtype=torch.float32),
                                  b[key]["distribution.std_param"] * torch.tensor(new, dtype=torch.float32), atol=1e-6)
        assert torch.allclose(ya, yb, atol=1e-5)


@pytest.mark.parametrize("direction", ["climb", "descend"])
def test_jitter_pyramid_stairs(direction):
    """Every riser within ±30 % of the level's nominal and every tread within 50-200 mm; vertical faces at each riser
    along +x and +y; the spawn on the pit floor (climb) or the top (descend), with no step within the standing
    robot's reach at any heading (the platform's half-width exceeds the footprint's radius)."""
    for difficulty_row in (0, 9):
        cfg = terrain.JitterPyramidStairsCfg(size=(1.6, 1.6), direction=direction)
        gen, z_at = _stair_model({"j": cfg}, 10, seed=1)
        rises, treads = cfg.last                      # the last row built (row 9)
        nom = 0.005 + (9 + 0.0) / 10 * 0.055
        assert all(0.7 * nom - 1e-9 <= r <= 1.3 * 0.060 + 1e-9 for r in rises)
        assert all(0.05 - 1e-12 <= t <= 0.20 + 1e-12 for t in treads)
        assert 0.20 + treads.sum() <= 0.70 + 1e-9
        half = 0.20 + np.concatenate([[0.0], np.cumsum(treads)])
        tops = np.cumsum(rises) if direction == "climb" else np.cumsum(rises[::-1])[::-1]
        for axis in ("x", "y"):
            for i, a in enumerate(half):
                pt = (lambda d: (0.8 + a + d, 0.8)) if axis == "x" else (lambda d: (0.8, 0.8 + a + d))
                z_out, z_in = z_at(9, 0, *pt(0.0015)), z_at(9, 0, *pt(-0.0015))
                if direction == "climb":
                    assert z_out == pytest.approx(tops[i], abs=1e-4)
                    assert z_in == pytest.approx(0.0 if i == 0 else tops[i - 1], abs=1e-4)
                else:
                    assert z_in == pytest.approx(tops[i], abs=1e-4)
                    assert z_out == pytest.approx(tops[i + 1] if i + 1 < len(tops) else 0.0, abs=1e-4)
        o = gen.terrain_origins[9, 0] - gen._get_sub_terrain_position(9, 0)
        assert o == pytest.approx([0.8, 0.8, 0.0 if direction == "climb" else tops[0]], abs=1e-9)
        assert np.hypot(0.0683 + 0.005, 0.0476 + 0.005) < 0.20 and np.hypot(0.11, 0.04) < 0.20  # feet, rear knees
        break


def test_stairs_mix():
    subs, b2 = terrain.stairs_sub_terrains(), terrain.rough_sub_terrains()
    assert abs(sum(c.proportion for c in subs.values()) - 1.0) < 1e-9
    assert "stairs_up" not in subs and abs(subs["jstairs_climb"].proportion - 0.30) < 1e-9
    assert abs(subs["jstairs_descend"].proportion - 0.20) < 1e-9
    assert abs(subs["low_poly"].proportion / subs["raised_relief"].proportion - 0.25 / 0.15) < 1e-9
    assert len(subs) == len(b2)


def test_jitter_stairs_record_where_they_end():
    cfg = terrain.JitterPyramidStairsCfg(size=(1.6, 1.6), direction="climb")
    _stair_model({"j": cfg}, 10, seed=2)
    assert len(cfg.extents) == 10
    assert cfg.extents[-1] == pytest.approx(0.20 + cfg.last[1].sum())
    assert all(0.20 < e <= 0.70 + 1e-9 for e in cfg.extents)


def test_curriculum_on_stair_tiles_needs_the_exit():
    """With stair_exit, a stair tile promotes only an env that got off the stairs (however little it walked) and
    demotes one that fell or never left the platform; other tiles keep the path rule."""
    # stairs: out / circled far but stayed in (path 2 m) / fell / stayed near the spawn / moved out a ring; flat: walked
    env, calls = _curriculum_env([0.3, 2.0, 0.9, 0.9, 0.9, 0.9], [1.2] * 6, [False, False, True, False, False, False])
    env.scene.terrain.terrain_types = torch.tensor([1, 1, 1, 1, 1, 0])
    env._sim_stairs = torch.tensor([False, True])
    env.sim_stair_exit = torch.tensor([True, False, False, False, False, False])
    env.sim_cheb_max = torch.tensor([0.7, 0.25, 0.4, 0.1, 0.45, 0.9])
    terrain_levels_path(env, torch.arange(6), promote_m=0.8)
    assert calls["up"].tolist() == [True, False, False, False, False, True]
    assert calls["down"].tolist() == [False, True, True, True, False, False]


def test_narrow_flights_record_their_geometry():
    """NarrowFlightCfg: per tile 4-5 risers scaling with the row (10 -> 35 mm, ±15 %), treads 45-80 mm, a width
    0.17-0.30 m and a landing 0.20-0.40 m; the drops at the sides and past the landing; each tile's flight recorded
    relative to its spawn in generation order."""
    cfg = terrain.NarrowFlightCfg(size=(1.6, 1.6))
    gen, z_at = _stair_model({"narrow": cfg}, 10, seed=4)
    f = cfg.flights[-10:]
    for row, x in enumerate(f):
        h = 0.010 + (row + 0.5) / 10 * 0.025
        assert len(x["rises"]) in (4, 5)
        assert all(0.85 * (h - 0.0025) - 1e-9 <= r <= 1.15 * (h + 0.0025) + 1e-9 for r in x["rises"])
        assert all(0.045 - 1e-9 <= t <= 0.080 + 1e-9 for t in x["treads"])
        assert 0.17 <= x["width"] <= 0.30 and 0.20 <= x["landing"] <= 0.40
        assert x["x_first"] == pytest.approx(0.15)
        assert x["x_last"] == pytest.approx(0.15 + sum(x["treads"]))
        o = gen.terrain_origins[row, 0] - gen._get_sub_terrain_position(row, 0)
        top = sum(x["rises"])
        assert z_at(row, 0, o[0] + x["x_last"] + 0.01, y=0.8) == pytest.approx(top, abs=1e-4)
        assert z_at(row, 0, o[0] + x["x_last"] + 0.01, y=0.8 + x["width"] / 2 + 0.005) == pytest.approx(0.0, abs=1e-4)
        assert z_at(row, 0, o[0] + x["x_last"] + x["landing"] + 0.01, y=0.8) == pytest.approx(0.0, abs=1e-4)


def test_narrow_riser_range_option():
    """--narrow-risers: ``stairs_narrow_sub_terrains(riser_range=)`` (via ``TerrainOptions.narrow_risers``) sets the
    flights' nominal riser range; the generated rows follow it."""
    from mp4_sim.gpu.terrain_task import TerrainOptions, sub_terrains
    assert sub_terrains(TerrainOptions(terrain="stairs_narrow"))["narrow"].riser_range == (0.010, 0.035)
    cfg = sub_terrains(TerrainOptions(terrain="stairs_narrow", narrow_risers=(0.020, 0.040)))["narrow"]
    assert cfg.riser_range == (0.020, 0.040)
    _stair_model({"narrow": cfg}, 10, seed=4)
    for row, x in enumerate(cfg.flights[-10:]):
        h = 0.020 + (row + 0.5) / 10 * 0.020
        assert all(0.85 * (h - 0.002) - 1e-9 <= r <= 1.15 * (h + 0.002) + 1e-9 for r in x["rises"])


@needs_gpu
def test_narrow_task_commands_spawn_exits_and_cross_track():
    """--terrain stairs_narrow with the robot's heading law: on the flights every env walks forward (not standing) holding the
    heading it spawned with (within ±2° of the flight), at most ±heading_wz of correction; a foot off a side ends
    the episode as a failure, all four feet on the landing as a success; the critic sees the cross-track drift."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.lab_api.math import wrap_to_pi

    from mp4_sim.gpu import terrain_task as tt
    opt = tt.TerrainOptions(arm="stock", num_envs=128, terrain="stairs_narrow", noisy=False, randomise=False, heading="loop",
                            stair_exit=True, heading_law="pi", heading_wz=0.4, cross_track=-1.0, max_init_level=9)
    cfg = tt.make_terrain_env_cfg(opt)
    assert list(cfg.observations["critic"].terms)[-1] == "track_errors"
    assert {"stair_exit", "side_exit"} <= set(cfg.terminations) and "cross_track" in cfg.rewards
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    n = env.num_envs
    zero = torch.zeros(n, 12, device="cuda:0")
    for _ in range(3):
        env.step(zero)
    t = env.scene.terrain
    col = list(tt.sub_terrains(opt)).index("narrow")
    nar = t.terrain_types == col
    assert int(nar.sum()) >= 8
    cmd = env.command_manager.get_term("twist")
    c = env.command_manager.get_command("twist")
    assert bool(cmd.is_heading_env[nar].all()) and not bool(cmd.is_standing_env[nar].any())
    assert bool(((c[nar, 0] >= 0.06) & (c[nar, 0] <= 0.12)).all())
    assert float(c[:, 2][cmd.is_heading_env].abs().max()) <= 0.4 + 1e-6
    robot = env.scene["robot"]
    assert float(wrap_to_pi(robot.data.heading_w[nar]).abs().max()) <= math.radians(2) + 0.02
    assert float(wrap_to_pi(cmd.heading_target[nar] - robot.data.heading_w[nar]).abs().max()) <= 0.02
    o = env.scene.env_origins
    assert float((robot.data.root_link_pos_w[nar, 1] - o[nar, 1]).abs().max()) <= 0.011
    # a side exit: move the narrow envs' bases sideways past the flight (a failure); the others stay
    ids = nar.nonzero().flatten()
    pose = torch.cat([robot.data.root_link_pos_w, robot.data.root_link_quat_w], -1)[ids].clone()
    pose[:, 1] += 0.20
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()
    side = tt.side_exit(env, tuple(tt.sub_terrains(opt).values()))
    assert bool(side[nar].all()) and not bool(side[~nar].any())
    # the landing: all four feet past the last riser
    tab = env._sim_flight[t.terrain_levels, t.terrain_types]
    pose[:, 1] -= 0.20
    pose[:, 0] = o[ids, 0] + tab[ids, 1] + 0.12
    pose[:, 2] += 0.2
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    env.sim.forward()
    ex = tt.stair_exit(env, tuple(tt.sub_terrains(opt).values()), opt.tile)
    assert bool(ex[nar].all())
    env.close()


def test_cross_track_clipped_saturates_and_ignores_standing():
    """CrossTrackClipped: |drift| / clip, at most 1; a standing command costs nothing and re-anchors."""
    from mp4_sim.gpu.terrain_task import CrossTrackClipped
    n = 3
    data = SimpleNamespace(root_link_pos_w=torch.zeros(n, 3), root_link_quat_w=torch.tensor([[1.0, 0, 0, 0]] * n))
    cmd = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02, scene={"robot": SimpleNamespace(data=data)},
                          command_manager=SimpleNamespace(get_command=lambda name: cmd))
    term = CrossTrackClipped(None, env)
    term(env, clip=0.10)
    data.root_link_pos_w = torch.tensor([[0.0, 0.03, 0.0], [0.0, 0.5, 0.0], [0.0, 0.5, 0.0]])
    r = term(env, clip=0.10)
    assert r.tolist() == pytest.approx([0.3, 1.0, 0.0], abs=1e-6)
    assert env.sim_cross_track[2].abs().max() == 0.0


@needs_gpu
def test_cross_track_resume_widens_the_critic_only(tmp_path, init_policy):
    """--cross-track --resume from a run without it: the critic's first layer gains 3 zero columns (track_errors,
    last), so the value is unchanged; the actor is untouched; the config records the weight."""
    import json
    _train("--init", init_policy, "--iters", 1, "--out", tmp_path / "a")
    src = next((tmp_path / "a").glob("*/model_0.pt"))
    _train("--resume", src, "--iters", 0, "--cross-track", -0.5, "--out", tmp_path / "b")
    run = next((tmp_path / "b").glob("*"))
    a = torch.load(src, weights_only=False, map_location="cpu")
    b = torch.load(run / "resume_critic_widened.pt", weights_only=False, map_location="cpu")
    w0, w1 = a["critic_state_dict"]["mlp.0.weight"], b["critic_state_dict"]["mlp.0.weight"]
    assert w1.shape[1] == w0.shape[1] + 3
    assert torch.equal(w1[:, :w0.shape[1]], w0) and not w1[:, w0.shape[1]:].any()
    assert all(torch.equal(a["actor_state_dict"][k], b["actor_state_dict"][k]) for k in a["actor_state_dict"])
    cfg = json.loads((run / "config.json").read_text())
    assert cfg["task"]["options"]["cross_track"] == -0.5


@needs_gpu
def test_yaw_catch_is_a_yaw_torque_after_its_start():
    """YawCatch: nothing before start_step; then pulses of |torque| in range about the world z axis only (no force, no
    roll or pitch torque) on the base."""
    from mjlab.envs import ManagerBasedRlEnv

    from mp4_sim.gpu import terrain_task as tt
    opt = tt.TerrainOptions(arm="stock", num_envs=32, terrain="flat", noisy=False, randomise=False,
                            yaw_catch=(0.10, 0.12), yaw_catch_start=10)
    cfg = tt.make_terrain_env_cfg(opt)
    cfg.events["yaw_catch"].params["cooldown_s"] = (0.0, 0.05)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    env.reset()
    robot = env.scene["robot"]
    base = robot.body_names.index("base")

    def xfrc():
        return env.sim.data.xfrc_applied[:, robot.indexing.body_ids[base]].clone()
    zero = torch.zeros(32, 12, device="cuda:0")
    for _ in range(5):
        env.step(zero)
    assert float(xfrc().abs().max()) == 0.0
    seen = torch.zeros(32, dtype=torch.bool, device="cuda:0")
    for _ in range(20):
        env.step(zero)
        w = xfrc()
        assert float(w[:, :5].abs().max()) == 0.0                           # force and roll/pitch torque
        tz = w[:, 5].abs()
        assert bool(((tz == 0) | ((tz >= 0.10 - 1e-6) & (tz <= 0.12 + 1e-6))).all())
        seen |= tz > 0
    assert int(seen.sum()) >= 16
    env.close()


def test_stairs_built_from_parameters():
    """--terrain stairs_built: one flight per riser, each ``stair_steps`` risers ``stair_tread`` apart, the width and
    landing applied; a single riser keeps the column name "built"."""
    from mp4_sim.gpu.terrain_task import TerrainOptions, sub_terrains
    one = sub_terrains(TerrainOptions(terrain="stairs_built", stair_width=0.20, stair_landing=0.25))
    assert list(one) == ["built"]
    cfg = one["built"]
    assert cfg.layout == ((0.025, 0.05),) * 4 and cfg.width == 0.20 and cfg.landing == 0.25 and cfg.tag == "built"
    two = sub_terrains(TerrainOptions(terrain="stairs_built", stair_risers=(0.02, 0.035), stair_tread=0.06,
                                      stair_steps=3))
    assert list(two) == ["built_20mm", "built_35mm"]
    _stair_model(two, 1)
    for tag, r in (("built_20mm", 0.02), ("built_35mm", 0.035)):
        L = terrain.STAIR_LAYOUTS[(tag, 0)]
        assert L["rises"] == [r] * 3 and L["treads"] == [0.06] * 2
