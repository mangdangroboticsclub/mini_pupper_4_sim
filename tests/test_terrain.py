"""Terrain (``envs.terrain``, ``envs.terrain_spawn``) on the CPU: the families and printed stairs, and the ground grid
the spawns read against MuJoCo's own rays."""
import numpy as np
import pytest

pytest.importorskip("mjlab")
import mujoco  # noqa: E402
from scene_cases import FAMILIES, ROUGH_FAMILIES  # noqa: E402
from test_envs import compile_scene  # noqa: E402

from mp4_sim.envs import terrain as tm  # noqa: E402
from mp4_sim.envs import terrain_spawn as ts  # noqa: E402

ALL = ("flat",) + ROUGH_FAMILIES + tuple(tm.PRINTED_STAIRS)


def test_family_columns_and_the_printed_stairs():
    cols = tm.family_columns(FAMILIES)
    fams = [tm.family_of(c) for c in cols]
    assert fams.count("flat") == len(cols) / 2
    assert set(fams) == set(ALL)
    four = cols["stairs4"]
    assert [tuple(r) for r in four.layout] == [(0.025, 0.05)] * 4 and four.width == 0.4
    assert len(cols["stairs8"].layout) == 8
    assert tm.family_columns(("low_poly",), relief_pitch=0.02)["low_poly"].pitch == 0.02


def test_ground_grid_matches_mujoco_rays_at_its_nodes():
    _, m = compile_scene(dict(num_envs=4, terrain="families", families=ALL, rows=4, difficulty_max=0.4,
                              max_init_level=0, sensors=("feet_contact",)))
    ground = ts.GroundGrid(m, "cpu")
    d = mujoco.MjData(m)
    d.qpos[2] = 50.0
    mujoco.mj_forward(m, d)
    rng = np.random.default_rng(0)
    nx, ny = ground.heights.shape
    g = np.array([-1], dtype=np.int32)
    h = ground.heights.cpu().numpy()
    origin = ground.origin.cpu().numpy()

    def ray(x, y):
        dist = mujoco.mj_ray(m, d, np.array([x, y, 3.0]), np.array([0, 0, -1.0]), None, 1, -1, g)
        return 3.0 - dist if dist >= 0 else 0.0

    near = [(0.0, 0.0), (2e-3, 0.0), (-2e-3, 0.0), (0.0, 2e-3), (0.0, -2e-3)]     # a node can lie on a box's edge
    err = []
    for i, j in zip(rng.integers(1, nx - 1, 2000), rng.integers(1, ny - 1, 2000)):
        x, y = origin + (i, j) * np.array(ground.pitch)
        err.append(min(abs(ray(x + dx, y + dy) - h[i, j]) for dx, dy in near))
    err = np.array(err)
    # a heightfield's cell is two triangles, the grid bilinear: they part on a steep cell, as at an obstacle's edge
    assert (err < 1e-3).mean() > 0.99 and err.max() < 0.02, (float((err < 1e-3).mean()), float(err.max()))
