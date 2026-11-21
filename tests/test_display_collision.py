"""The SKU1's optional display collision box (``robot.model.add_display_collision``, ``SceneOptions.display_collision``)."""
import math

import mujoco
import numpy as np
import pytest

from mp4_sim import robots
from mp4_sim.envs import contact
from mp4_sim.robot import model
from mp4_sim.robot.model import DISPLAY_GEOM, Legs

SKU1 = Legs(robot="sku1")


def _display_vertices():
    trimesh = pytest.importorskip("trimesh")
    d = robots.SKU1.dir / "parts"
    return np.vstack([trimesh.load(d / f"{n}.stl", force="mesh").vertices for n in ("base_display", "base_screen")])


def test_box_holds_the_display_meshes_and_is_tight():
    v = _display_vertices()
    pos, quat, half = model.display_box_frame("sku1")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    w = (v - pos) @ R.reshape(3, 3)
    assert np.all(np.abs(w) <= half + 1e-9)                       # every vertex inside
    assert np.all(np.abs(w).max(0) >= half - 2.1e-5)              # each face within 0.02 mm of a vertex
    assert np.allclose(2e3 * half, (40.84, 54.64, 10.10), atol=1e-6)
    assert np.prod(2 * half) < 0.5 * np.prod(np.ptp(v, 0))        # under half the axis-aligned bounds' volume


def test_default_off_and_sku2_refused():
    for robot in ("sku1", "sku2"):
        m = model.get_spec(legs=Legs(robot=robot)).compile()
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, DISPLAY_GEOM) == -1
    assert robots.SKU2.display_box is None
    with pytest.raises(ValueError, match="no display collision box"):
        model.get_spec(legs=Legs(), display_collision=True)


def test_box_in_the_model_with_the_base_box_contacts():
    m = model.get_spec(legs=SKU1, display_collision=True).compile()
    g = m.geom(DISPLAY_GEOM)
    base = m.geom([i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("base").id
                   and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX and i != g.id][0])
    assert m.geom_bodyid[g.id] == m.body("base").id and g.type[0] == mujoco.mjtGeom.mjGEOM_BOX
    for f in ("contype", "conaffinity", "priority", "condim", "friction", "solref", "solimp", "margin", "group"):
        assert np.array_equal(getattr(g, f), getattr(base, f)), f
    assert np.allclose(g.size, robots.SKU1_DISPLAY_BOX[2]) and np.allclose(g.pos, robots.SKU1_DISPLAY_BOX[0])
    assert math.isclose(2 * math.degrees(math.acos(g.quat[0])), 55.0, abs_tol=1e-9)
    bare = model.get_spec(legs=SKU1).compile()
    assert np.allclose(m.body_mass, bare.body_mass) and np.allclose(m.body_inertia, bare.body_inertia)
    assert m.ngeom == bare.ngeom + 1


def test_the_self_collision_options_treat_it_as_the_base():
    spec = model.get_spec(legs=SKU1, display_collision=True, self_collision=True)
    contact.getup_collisions(spec)
    m = spec.compile()
    g, base = m.geom(DISPLAY_GEOM), m.geom([i for i in range(m.ngeom) if m.geom_bodyid[i] == m.body("base").id
                                            and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX
                                            and m.geom(i).name != DISPLAY_GEOM][0])
    assert int(g.contype[0]) == int(base.contype[0]) and int(g.conaffinity[0]) == int(base.conaffinity[0])
    assert int(g.conaffinity[0]) & 4 and int(g.conaffinity[0]) & contact.BASE_BIT


def test_scene_option_reaches_the_robot():
    pytest.importorskip("mjlab")
    from mp4_sim.envs.scene import SceneOptions, scene_cfg
    assert SceneOptions().display_collision is False
    for on in (False, True):
        sc = scene_cfg(SceneOptions(num_envs=1, robot="sku1", display_collision=on))
        m = sc.scene.entities["robot"].spec_fn().compile()
        assert (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, DISPLAY_GEOM) >= 0) is on
