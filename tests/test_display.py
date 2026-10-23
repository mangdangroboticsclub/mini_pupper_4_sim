"""The screen overlay: in front of the base, visual only, and outside everything the physics is hashed from."""
import mujoco
import numpy as np
import pytest

from mp4_sim import paths
from mp4_sim.robot import display, sku2
from mp4_sim.robot.sku import resolve_sku

SKUS = ["sku_1", "sku_2"]


def _scene(sku):
    """The SKU's pupper.xml (ghost and all) with the screen added."""
    spec = mujoco.MjSpec.from_file(str(resolve_sku(sku).robot_xml))
    display.add_display(spec, sku)
    return spec.compile()


@pytest.mark.parametrize("sku", SKUS)
def test_display_adds_no_mass_and_no_contacts(sku):
    plain = mujoco.MjModel.from_xml_path(str(resolve_sku(sku).robot_xml))
    model = _scene(sku)
    np.testing.assert_array_equal(model.body_mass, plain.body_mass)
    np.testing.assert_array_equal(model.body_inertia, plain.body_inertia)
    for part in display.load(sku)["parts"]:
        geom = model.geom(part["mesh"])
        assert geom.contype[0] == 0 and geom.conaffinity[0] == 0
        assert model.body(geom.bodyid[0]).name == "base"


@pytest.mark.parametrize("sku", SKUS)
def test_display_is_on_the_front_of_the_base(sku):
    model = _scene(sku)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.body("base").id
    lo, hi = np.array(display.load(sku)["bounds_m"])
    assert lo[0] > 0.06 and abs(lo[1] + hi[1]) < 1e-3          # in front, centred left-right
    for part in display.load(sku)["parts"]:
        mesh = model.geom(part["mesh"])
        local = data.xmat[base].reshape(3, 3).T @ (data.geom_xpos[mesh.id] - data.xpos[base])
        assert np.all(local > lo - 1e-3) and np.all(local < hi + 1e-3)


def test_sku2_screen_sits_on_the_placeholder():
    info = display.load("sku_2")
    lo, hi = np.array(info["bounds_m"])
    np.testing.assert_allclose((lo + hi) / 2, info["placement"]["placeholder_centre_m"], atol=1e-6)


def test_sku2_spec_has_the_screen_only_with_visuals():
    with_visuals = sku2.get_spec(strip_visuals=False).compile()
    physics = sku2.get_spec().compile()
    assert mujoco.mj_name2id(with_visuals, mujoco.mjtObj.mjOBJ_GEOM, "display_bezel") >= 0
    assert mujoco.mj_name2id(physics, mujoco.mjtObj.mjOBJ_GEOM, "display_bezel") < 0
    np.testing.assert_array_equal(with_visuals.body_mass, physics.body_mass)


def test_display_sources_recorded():
    cad = paths.REPO / "cad" / "sku_1" / "display"
    for sku in SKUS:
        for name, digest in display.load(sku)["sources"].items():
            assert paths.sha256(cad / name) == digest, (sku, name)
