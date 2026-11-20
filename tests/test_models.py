"""The robot model (``robot.model``): the packaged models, their feet and stands, the compiled models pinned."""
import hashlib
import json
import math

import mujoco
import numpy as np
import pytest

from mp4_sim import paths, robots
from mp4_sim.calibration import Contract
from mp4_sim.envs.contact import CONTACT_SOLREF, THIGH_SHORT_M, getup_collisions, wrap_spec_fn
from mp4_sim.robot import model
from mp4_sim.robot.model import Legs, legs_meta, legs_of

FEET = (Legs(foot="sphere"), Legs(foot="mesh"))
WALKER = dict(masses_json="sku2_masses.json", self_collision=True, foot_solref=(0.01, 1.0))
# compiled-model hashes (MuJoCo 3.7.0), per foot: a walking plant and a get-up plant (contact edits), with and without
# the limits
HASHES = {
    "sphere": {"walker": "ae6537d38238a91a505ec45aa01d853cf4c0ed1c1b20084b7cc94cdcdc302748",
               "walker_limits": "3a47e1b8ffbd60ddb60f797b86ca23b526f3c1f95eef8cd879f15d8d0830e3a2",
               "getup": "7e53f1f4263cb98ce0fc3073da104697bcffab94ef2722b1da90dbbe3052c0fd"},
    "mesh": {"walker": "c0e2d94344c01a50df68d5eea03718c77728bdbe681c9247ad68ca7ce6b14bc8",
             "walker_limits": "f70da4f02220b15981830aa58dc39fcbbe80771ad1707221bf2cce3ab35b5cfa",
             "getup": "d2c68bce61128b6b1f8089fe0db78c9ba8956465f6d8ba413b8b8019211c5b9f"},
}
# SKU1 (the 413 g masses; "walker" names sku2_masses.json, which the SKU1 replaces by its own)
SKU1_HASHES = {
    "default": "5010fd3cc388a9b5f7962ec68ee06dd6ef4c89590ed02f7e10fe8b6769420ccd",
    "walker": "888600a740c6304dd5b30102f921d2c652959a8f5e7dbb6b8d15e84ef337ab73",
    "walker_mesh": "b5c4e3e1c1952b32ae57500f0ad2639ed306668eafc5c9c19f1b4da9b852325f",
}
MODEL_SHA256 = "455316472c9c059c7bb9e2d5236f87c59a6d58e42071c89507fcbe8bf83f3dcf"     # robot/assets/sku_2/pupper.xml
SKU1_SHA256 = "5ce84acc8ce53d1597d2f5df8c8ffc1ba3b56055e3f489c395cfc9211d2c2217"      # robot/assets/sku_1/pupper.xml
SKU1_SCREEN = {"base_display", "base_screen", "base_screen_face"}                     # the SKU1's visual-only screen
LEG_BIT = 8
PINNED = pytest.mark.skipif(mujoco.__version__ != "3.7.0", reason="hashes pinned with MuJoCo 3.7.0")


def model_hash(m: mujoco.MjModel, skip: tuple = ()) -> str:
    h = hashlib.sha256()
    for k in sorted(dir(m)):
        if k.startswith("_") or k in skip:
            continue
        try:
            v = getattr(m, k)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


@PINNED
@pytest.mark.parametrize("legs", FEET, ids=("sphere", "mesh"))
def test_compiled_models_are_pinned(legs):
    want = HASHES[legs.foot]
    for lg in (legs, Legs("new", legs.foot, robot="sku2"), legs_of({"legs": "sku_2", "foot": legs.foot})):
        assert model_hash(model.get_spec(**WALKER, legs=lg).compile()) == want["walker"]
        assert model_hash(model.get_spec(**WALKER, limits=True, legs=lg).compile()) == want["walker_limits"]
        g = wrap_spec_fn(lambda: model.get_spec(**WALKER, legs=lg), True, True, THIGH_SHORT_M, CONTACT_SOLREF)
        assert model_hash(g().compile()) == want["getup"]


@PINNED
def test_sku1_compiled_models_are_pinned():
    legs = Legs(robot="sku1")
    assert model_hash(model.get_spec(legs=legs).compile()) == SKU1_HASHES["default"]
    with pytest.warns(UserWarning, match="sku2_masses.json"):
        assert model_hash(model.get_spec(**WALKER, legs=legs).compile()) == SKU1_HASHES["walker"]
    with pytest.warns(UserWarning):
        m = model.get_spec(**WALKER, legs=Legs(foot="mesh", robot="sku1")).compile()
    assert model_hash(m) == SKU1_HASHES["walker_mesh"]


def test_packaged_model_builds_and_its_sources():
    assert paths.sha256(robots.SKU2.xml) == MODEL_SHA256
    for legs in FEET:
        for strip in (True, False):
            model.get_spec(strip_visuals=strip, legs=legs).compile()
    assert model.sources() == {"mp4_sim:robot/assets/sku_2/pupper.xml": MODEL_SHA256,
                               "mp4_sim:robot/assets/sku_2/measured/sku2_masses.json":
                                   paths.sha256(robots.SKU2.masses_path)}
    s1 = model.sources(Legs(robot="sku1", foot="mesh"))
    assert s1["mp4_sim:robot/assets/sku_1/pupper.xml"] == SKU1_SHA256
    assert "mp4_sim:robot/assets/sku_1/measured/sku1_masses_413g.json" in s1
    assert "mp4_sim:robot/assets/sku_1/parts/FR_foot.stl" in s1


def _table(m):
    out = {}
    for i in range(m.nbody):
        b = m.body(i)
        out["body", b.name] = (b.mass.tolist(), b.ipos.tolist(), b.inertia.tolist(), b.iquat.tolist(), b.pos.tolist(),
                               b.quat.tolist())
    for i in range(m.njnt):
        j = m.joint(i)
        out["joint", j.name] = (j.type.tolist(), j.axis.tolist(), j.pos.tolist(), j.range.tolist(), j.limited.tolist(),
                                j.damping.tolist(), j.armature.tolist(), j.frictionloss.tolist())
    for i in range(m.ngeom):
        g = m.geom(i)
        out["geom", g.name or f"{m.body(int(g.bodyid[0])).name}#{i}"] = (
            g.type.tolist(), g.size.tolist(), g.pos.tolist(), g.quat.tolist(), g.contype.tolist(),
            g.conaffinity.tolist(), g.priority.tolist(), g.solref.tolist(), g.friction.tolist())
    for i in range(m.nsite):
        out["site", m.site(i).name] = m.site(i).pos.tolist()
    return out


def test_the_feet_differ_only_in_the_foot_geoms():
    a, b = (model.get_spec(**WALKER, limits=True, legs=legs) for legs in FEET)
    getup_collisions(a)
    getup_collisions(b)
    ta, tb = _table(a.compile()), _table(b.compile())
    assert set(ta) == set(tb)
    assert {k for k in ta if ta[k] != tb[k]} == {("geom", f"{leg}_foot") for leg in model.LEGS}
    for leg in model.LEGS:
        assert ta["geom", f"{leg}_foot"][4:] == tb["geom", f"{leg}_foot"][4:]
        assert tb["site", f"{leg}_foot"] == pytest.approx([0.0422496, 0.0422496, 0.0])


def test_mesh_feet_and_the_sphere_bottom():
    f = model.mesh_feet()
    for leg in model.LEGS:
        ext = (f[leg]["bbox_max"] - f[leg]["bbox_min"]) * 1000
        assert ext == pytest.approx([19.26, 19.64, 6.0], abs=0.01)
        assert f[leg]["vertices"] == 2818 and f[leg]["faces"] == 5632
    sphere, mesh = (model.stand_geometry(L) for L in FEET)
    assert sphere["base_z_m"] == pytest.approx(mesh["base_z_m"], abs=1e-9)
    for leg in model.LEGS:
        assert sphere["foot_site_clearance_m"][leg] == pytest.approx(mesh["foot_site_clearance_m"][leg], abs=1e-9)
        assert sphere["foot_lowest_z_m"][leg] == pytest.approx(0.0, abs=1e-7)
    assert model.stand_geometry(Legs(foot_radius_m=0.003))["base_z_m"] == pytest.approx(sphere["base_z_m"], abs=1e-9)


def test_stand_heights():
    g = model.stand_geometry()
    assert g["base_z_m"] == pytest.approx(0.075162, abs=2e-6)
    assert all(v == pytest.approx(0.072206, abs=2e-6) for v in g["hip_axis_height_m"].values())
    assert model.init_base_z() == pytest.approx(0.0747065, abs=1e-7)
    assert model.foot_site_clearance() == pytest.approx(0.00411, abs=1e-5)
    assert model.robot_z_shift() == 0.0


def _stand_sim(legs, seconds=2.0):
    """A walking plant on a floor, position servos (kp 8, kv 0.05, 0.8 N m) holding the calibrated stand from the spawn
    height."""
    c = legs.contract()
    stand = dict(zip(c.joint_order, c.stand_joint_rad))
    spec = model.get_spec(**{**WALKER, "masses_json": None}, legs=legs)
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.05], contype=1, conaffinity=1)
    for leg in model.LEGS:
        for name, trn, target in ((f"{leg}_hip", mujoco.mjtTrn.mjTRN_JOINT, f"{leg}_hip"),
                                  (f"{leg}_thigh", mujoco.mjtTrn.mjTRN_JOINT, f"{leg}_thigh"),
                                  (f"{leg}_calf", mujoco.mjtTrn.mjTRN_TENDON, f"{leg}_knee_drive")):
            a = spec.add_actuator(name=name, target=target, trntype=trn)
            a.set_to_position(kp=8.0, kv=0.05)
            a.forcerange, a.forcelimited = [-0.8, 0.8], 1
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[2], d.qpos[3] = model.init_base_z(legs), 1.0
    for name, v in stand.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = v
    mujoco.mj_forward(m, d)
    for leg in model.LEGS:
        d.ctrl[m.actuator(f"{leg}_hip").id] = stand[f"{leg}_hip"]
        d.ctrl[m.actuator(f"{leg}_thigh").id] = stand[f"{leg}_thigh"]
        d.ctrl[m.actuator(f"{leg}_calf").id] = d.ten_length[m.tendon(f"{leg}_knee_drive").id]
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    tilt = math.degrees(math.acos(min(1.0, d.xmat[m.body("base").id][8])))
    return float(d.qpos[2]), tilt, d


@pytest.mark.parametrize("legs", FEET + (Legs(robot="sku1"),), ids=("sphere", "mesh", "sku1"))
def test_stand_for_2_s_on_flat_ground(legs):
    z, tilt, d = _stand_sim(legs)
    assert tilt < 0.5
    assert z == pytest.approx(model.stand_geometry(legs)["base_z_m"], abs=0.001)
    assert d.ncon == 4                                                               # the four feet, nothing else


def test_options_and_meta():
    with pytest.raises(ValueError):
        Legs("other")
    with pytest.raises(ValueError):
        Legs(foot="cube")
    with pytest.raises(ValueError):
        Legs(foot_radius_m=0.03)
    with pytest.raises(ValueError):
        Legs(robot="sku3")
    assert legs_of(None) == legs_of({}) == legs_of({"legs": None, "foot": None}) == Legs() == Legs("new")
    assert legs_of(None, robot="sku1") == Legs(robot="sku1")
    for legs in FEET + (Legs(foot_radius_m=0.006), Legs(robot="sku1")):
        assert legs_of({"legs": legs_meta(legs)}) == legs
    m = legs_meta()
    assert (m["legs"], m["model"], m["model_sha256"]) == ("sku_2", "robot/assets/sku_2/pupper.xml", MODEL_SHA256)
    assert "robot" not in m and "robot_provisional" not in m
    m1 = legs_meta(Legs(robot="sku1"))
    assert m1["robot"] == "sku1" and m1["robot_provisional"] and m1["model_sha256"] == SKU1_SHA256
    assert m1["model"] == "robot/assets/sku_1/pupper.xml"


def collision_table(m):
    return [(m.geom(g).name, m.body(m.geom_bodyid[g]).name, int(m.geom_type[g]), m.geom_size[g].tolist(),
             m.geom_pos[g].tolist(), int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
            for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]


def test_sku2_masses_and_collisions():
    for kw in ({}, WALKER):
        m = model.get_spec(**kw, legs=Legs()).compile()
        assert abs(m.body_mass.sum() - 0.6305) < 1e-12
        col = collision_table(m)
        assert len(col) == 18 and sum(1 for c in col if c[1] == "base") == 2      # the two base boxes


def test_sku1_builds_with_the_measured_masses():
    legs = Legs(robot="sku1")
    assert legs.xml == robots.SKU1.xml
    j = json.loads(robots.SKU1.masses_path.read_text())
    assert j["robot"] == "sku1" and j["total_kg"] == 0.4134
    for masses_json in (None, "sku1_masses_413g.json"):
        m = model.get_spec(masses_json=masses_json, self_collision=True, legs=legs).compile()
        assert abs(m.body_mass.sum() - 0.4134) < 1e-12, masses_json
    p = str(robots.SKU2.masses_path)                     # a path loads as given
    assert abs(model.get_spec(masses_json=p, legs=legs).compile().body_mass.sum() - 0.6305) < 1e-9
    m = model.get_spec(legs=legs).compile()
    col = collision_table(m)
    assert len(col) == 17 and [c[3] for c in col if c[1] == "base"] == [[0.072, 0.04, 0.0135]]
    # the whole robot's centre of mass is the weighed one at the calibrated stand: x -4.30 mm, y +1.74 mm, 65.93 mm up
    d = model.stand_data(m, "sku1")
    com = d.subtree_com[m.body("base").id] - d.xpos[m.body("base").id]
    assert abs(com[0] + 0.00430) < 0.0002 and abs(com[1] - 0.00174) < 0.0002
    assert abs(com[2] + model.stand_geometry(legs)["base_z_m"] - 0.06593) < 0.0003
    g1, g2 = model.stand_geometry(legs), model.stand_geometry(Legs())
    for leg in model.LEGS:                             # the SKU2's legs: the same hip-axis height, a shorter body
        assert abs(g1["hip_axis_height_m"][leg] - g2["hip_axis_height_m"][leg]) < 1e-6
    z1, z2 = model.init_base_z(legs), model.init_base_z(Legs())
    assert z1 < z2 and abs(model.robot_z_shift(legs) - (z1 - z2)) < 1e-15
    fr, rr = (m.body(f"{x}_hip").pos[0] for x in ("FR", "RR"))
    assert abs(fr - rr - 0.09708) < 1e-4


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_visuals_are_visual_only(robot):
    """With the visuals kept: the same masses and collisions; the SKU1's screen is there and contact-free; the SKU2's
    accessories are not packaged and stay removed."""
    legs = Legs(robot=robot)
    vis = model.get_spec(strip_visuals=False, legs=legs).compile()
    bare = model.get_spec(legs=legs).compile()
    assert np.array_equal(vis.body_mass, bare.body_mass) and np.array_equal(vis.body_inertia, bare.body_inertia)
    assert collision_table(vis) == collision_table(bare)
    names = {vis.mesh(int(vis.geom_dataid[g])).name for g in range(vis.ngeom) if vis.geom_type[g] == 7}
    wanted = set() if robot == "sku2" else SKU1_SCREEN
    assert wanted <= names and not names & (set(model.ACCESSORY_MESHES) - wanted)
    for g in range(vis.ngeom):
        if vis.geom_type[g] == 7 and vis.mesh(int(vis.geom_dataid[g])).name in wanted:
            assert vis.geom_contype[g] == 0 and vis.geom_conaffinity[g] == 0


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_accessories_stay_contact_free_with_the_collision_bits(robot):
    spec = model.get_spec(strip_visuals=False, self_collision=True, legs=Legs(robot=robot))
    getup_collisions(spec)
    m = spec.compile()
    for g in range(m.ngeom):
        if m.geom_type[g] == 7 and m.mesh(int(m.geom_dataid[g])).name in SKU1_SCREEN:
            assert m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def test_the_stand_is_the_robots_calibration():
    for robot in robots.ROBOTS:
        c = Contract.load(robot=robot)
        m = model.get_spec(legs=Legs(robot=robot)).compile()
        d = model.stand_data(m, robot)
        for name, v in zip(c.joint_order, c.stand_joint_rad):
            assert d.qpos[m.jnt_qposadr[m.joint(name).id]] == v
