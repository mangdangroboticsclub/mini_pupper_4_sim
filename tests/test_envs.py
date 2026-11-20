"""The simulation scene (``envs.scene``) on the CPU: the compiled scenes pinned, the sensors, the events and their
order, the attachment, the centre-of-mass height, the contact edits and the measured noise."""
import itertools
import random
import uuid

import mujoco
import numpy as np
import pytest

pytest.importorskip("mjlab")
import torch  # noqa: E402
from scene_cases import CASES  # noqa: E402
from test_models import PINNED, model_hash  # noqa: E402

from mp4_sim import robots  # noqa: E402
from mp4_sim.envs import com_height as ch  # noqa: E402
from mp4_sim.envs import contact  # noqa: E402
from mp4_sim.envs import scene as S  # noqa: E402
from mp4_sim.envs.attachment import HOOD_TOP_Z, Attachment, add_cylinder  # noqa: E402
from mp4_sim.envs.noise import noise_profile  # noqa: E402
from mp4_sim.robot import model  # noqa: E402

# compiled scenes (robot, terrain, sensors; MuJoCo 3.7.0, mjlab 1.3.0): the walking and get-up plants of the released
# policies' environments; the hash leaves out names_map (it follows the terrain's random mesh names)
SCENE_HASHES = {
    "plane": "9f9dce6a5fe55712b61785c79b31004ac319e55a2d822e39fac922457126a1a9",
    "rough": "3e685ac0fed7e40fd1356bd8c849e9fb363a090665d6d0fdb6cb09a3632ecd35",
    "flat_limits_mesh": "090040607246ddfe914131a490d6806694e7c50c4c537179fc30175e86f86f1c",
    "stairs": "921ec6a38a6636a7d9e12cc83f08afbe7c5f7509201cc97a65874f5f51c99d34",
    "getup_lidar": "84f9a667c73829514135b916af53d4d227f08f9c902a000e518c73bb154509e6",
    "getup_terrain": "5285d9d092322ba5efe25792e60483e209e98ce416b64350a893d4aafab7618f",
    "sku1_getup_com": "08f3b0cf59f022e013bdef1a0eacb9f67bfafb3bed7e8a80c6449785d647d2d4",
}


def compile_scene(opt):
    from mjlab.scene import Scene
    count = itertools.count()
    uuid4 = uuid.uuid4
    uuid.uuid4 = lambda: uuid.UUID(int=next(count))
    try:
        sc = S.scene_cfg(S.SceneOptions(**opt))
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        return sc, Scene(sc.scene, device="cpu").compile()
    finally:
        uuid.uuid4 = uuid4


def scene_hash(m):
    return model_hash(m, skip=("names_map",))


@PINNED
@pytest.mark.parametrize("case", sorted(CASES))
def test_compiled_scenes_are_pinned(case):
    _, m = compile_scene(CASES[case])
    assert scene_hash(m) == SCENE_HASHES[case]


def test_sensors_by_terrain_and_by_name():
    assert [s.name for s in S.scene_cfg(S.SceneOptions(num_envs=1)).scene.sensors] == list(S.PLANE_SENSORS)
    sc = S.scene_cfg(S.SceneOptions(num_envs=1, terrain="flat", rows=1))
    assert [s.name for s in sc.scene.sensors] == list(S.TERRAIN_SENSORS)
    sc = S.scene_cfg(S.SceneOptions(num_envs=1, terrain="flat", rows=1, sensors=("feet_contact", "body_scan")))
    assert [s.name for s in sc.scene.sensors] == ["feet_contact", "body_scan"]
    with pytest.raises(ValueError):
        S.sensor_cfg("lidar")
    with pytest.raises(ValueError):
        S.scene_cfg(S.SceneOptions(terrain="moon"))


def test_terrain_extent_and_rows():
    sc = S.scene_cfg(S.SceneOptions(num_envs=1, terrain="rough", rows=10))
    assert sc.rows == 10 and sc.half_x == 10 * 1.6 / 2 + S.BORDER
    assert sc.half_y == len(sc.sub_terrains) * 1.6 / 2 + S.BORDER
    sc = S.scene_cfg(S.SceneOptions(num_envs=1, terrain="square_walls", rows=10))
    assert sc.rows == 1
    plane = S.scene_cfg(S.SceneOptions(num_envs=1))
    assert plane.sub_terrains is None and plane.half_x is None and plane.rows == 1
    assert plane.foot_site_clearance == pytest.approx(0.00411, abs=1e-5)
    d = S.stair_layout_file("four_steps")
    assert d["steps_mm"][0] == [25, 50] and S.stair_layout_file("eight_steps.json")["width_mm"] == 400


def test_the_plant_events_and_their_order():
    sc = S.scene_cfg(S.SceneOptions(num_envs=1))
    assert list(sc.events) == ["bam_init", "foot_friction", "base_com"] and not sc.late_events
    assert all(e.mode == "startup" for e in sc.events.values())
    assert sc.events["foot_friction"].params["ranges"] == (0.4, 1.2)
    assert sc.events["base_com"].params["ranges"] == S.BASE_COM_RANGES
    assert list(S.scene_cfg(S.SceneOptions(num_envs=1, randomise=False)).events) == ["bam_init"]
    sc = S.scene_cfg(S.SceneOptions(**CASES["sku1_getup_com"]))
    assert list(sc.events) == ["bam_init", "foot_friction", "base_com", "base_mass"]
    assert list(sc.late_events) == ["attachment", ch.EVENT]
    assert sc.late_events["attachment"].params["top_z"] == robots.SKU1.top_z_m
    p = sc.late_events[ch.EVENT].params
    assert p["bodies"] == ch.COM_BODIES and p["axes"] == (2, 1, 1, 1, 1) and p["signs"] == (1.0,) * 5
    sc2 = S.scene_cfg(S.SceneOptions(**CASES["getup_lidar"]))
    assert set(sc2.late_events["attachment"].params) == {"att"}                  # SKU2: the hood's own top
    first = {"noise": object(), "reset_base": object()}
    middle = {"foot_softness": object(), "getup_start": object()}
    order = list(S.ordered_events(sc, first, middle))
    assert order == ["noise", "reset_base", "bam_init", "foot_friction", "base_com", "base_mass", "foot_softness",
                     "getup_start", "attachment", ch.EVENT]
    with pytest.raises(ValueError, match="twice"):
        S.ordered_events(sc, {"bam_init": object()})


def test_contact_edits():
    def bits(m):
        return [(int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom)]
    plain = model.get_spec(self_collision=True)
    m0 = plain.compile()
    assert not any(a & contact.LEG_BIT for _, a in bits(m0))
    sc = S.scene_cfg(S.SceneOptions(num_envs=1, self_collision=True, collisions=True))
    m1 = sc.scene.entities["robot"].spec_fn().compile()
    assert any(a & contact.LEG_BIT for _, a in bits(m1))
    assert any(c & contact.BASE_BIT for c, _ in bits(m1))
    # with the limits: no thigh <-> base
    sc = S.scene_cfg(S.SceneOptions(num_envs=1, self_collision=True, collisions=True, limits=True))
    m2 = sc.scene.entities["robot"].spec_fn().compile()
    thighs = [g for g in range(m2.ngeom) if m2.geom(g).name.endswith("_thigh_collision")]
    assert thighs and not any(m2.geom_contype[g] & contact.BASE_BIT for g in thighs)
    spec = model.get_spec()
    k = contact.stiff_contacts(spec)
    m3 = spec.compile()
    assert k > 0 and all(m3.geom_priority[m3.geom(f"{leg}_foot").id] == 1 for leg in model.LEGS)
    assert m3.geom_priority[m3.geom("FR_thigh_collision").id] == 2


def test_attachment_y_range_and_presence():
    assert Attachment().y_mm == (0.0, 0.0)
    with pytest.raises(ValueError):
        Attachment(y_mm=(5.0, -5.0))
    for bad in (dict(height_mm=(-1.0, 5.0)), dict(mass_g=(-1.0, 5.0)), dict(x_mm=(10.0, -20.0))):
        with pytest.raises(ValueError):
            Attachment(**bad)
    att = Attachment.parse("35,19,-0.005,47.2")
    assert att.x_mm == (-5.0, -5.0) and att.mass_g == (47.2, 47.2)
    spec = model.get_spec(legs=model.Legs(robot="sku1"))
    add_cylinder(spec, Attachment((35.0, 35.0), (47.2, 47.2), (-5.0, -5.0), 19.0, y_mm=(2.0, 2.0)), 0.0135)
    g = spec.compile().geom("attachment")
    assert np.allclose(g.pos, [-0.005, 0.002, 0.0135 + 0.0175]) and g.type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert robots.SKU2.top_z_m == HOOD_TOP_Z


def test_com_height_range_and_the_shared_draw():
    n = ch.nominal(None, model.Legs(robot="sku1"))
    assert abs(n["height_m"] - 0.0658) < 0.0003 and abs(n["total_kg"] - 0.4134) < 1e-9
    k = n["total_kg"] / n["moved_kg"]
    lo, hi = ch.body_range((50.0, 66.0), None, model.Legs(robot="sku1"))
    assert abs(lo - (0.050 - n["height_m"]) * k) < 1e-12 and abs(hi - (0.066 - n["height_m"]) * k) < 1e-12
    with pytest.raises(ValueError):
        ch.body_range((70.0, 60.0), None, model.Legs())

    class Robot:
        indexing = type("I", (), {"body_ids": torch.arange(14)})()

        def find_bodies(self, name):
            return [["world", "base", "FR_hip", "x", "y", "FL_hip", "x", "y", "RR_hip", "x", "y", "RL_hip"].index(
                name)], [name]

    class Env:
        num_envs, device = 64, "cpu"
        scene = {"robot": Robot()}
        sim = type("Sim", (), {"model": type("M", (), {"body_ipos": torch.zeros(64, 14, 3)})()})()

    env = Env()
    ch.shift_com_height(env, None, (-0.01, 0.0), ch.COM_BODIES, (2, 1, 1, 1, 1), (1.0,) * 5)
    ip = env.sim.model.body_ipos
    d = env.com_height_draw
    assert torch.all(d <= 0) and torch.all(d >= -0.01) and d.std() > 0
    assert torch.equal(ip[:, 1, 2], d) and all(torch.equal(ip[:, i, 1], d) for i in (2, 5, 8, 11))
    assert ch.event_cfg(S.SceneOptions()) is None


def test_measured_noise():
    prof = noise_profile()
    assert set(prof) == {"ang_vel", "gravity", "pos", "vel"}
    assert len(prof["pos"]) == 12 and len(prof["ang_vel"]) == 3
    assert noise_profile(robot="sku1") == prof                   # provisional: the SKU2's
    with pytest.raises(ValueError):
        noise_profile("n2")
