"""The calibration (``calibration``): firmware units, coordinates, each robot's file."""
import json

import numpy as np
import pytest

from mp4_sim import robots
from mp4_sim.calibration import (
    CONTRACT_JSON,
    DD0,
    Contract,
    contract_path,
    contract_tendon_coefs,
    leg_set_name,
)

C = Contract.load()


def test_default_is_sku2_and_a_robot_selects_its_file():
    assert CONTRACT_JSON == robots.SKU2.contract_json == contract_path() == contract_path("sku2")
    assert contract_path("sku1") == contract_path(robots.SKU1) == robots.SKU1.contract_json
    assert np.array_equal(Contract.load(robot="sku2").stand_dd, C.stand_dd)
    c1 = Contract.load(robot="sku1")
    assert not np.array_equal(c1.stand_dd, C.stand_dd)
    assert np.array_equal(Contract.load(robots.SKU1.contract_json).stand_dd, c1.stand_dd)
    with pytest.raises(ValueError):
        contract_path("sku3")


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_stand_ticks_and_dd_match_the_stand(robot):
    """servo_write(0 + offset) gives the calibration's stand ticks and dd."""
    c = Contract.load(robot=robot)
    ticks = c.ticks_from_servo_deg(np.zeros(12))
    assert ticks.tolist() == c.stand_ticks.tolist()
    assert c.dd_from_ticks(ticks).tolist() == c.stand_dd.tolist()


def test_dd_inverse_is_exact_on_the_tick_grid():
    """A dd produced from a tick reads back within one tick of the angle that produced it."""
    rng = np.random.default_rng(1)
    s = rng.uniform(-40, 40, (200, 12))
    dd = C.dd_from_ticks(C.ticks_from_servo_deg(s))
    back = C.servo_deg_from_dd(dd)
    assert np.abs(back - s).max() < 3100 / 10240 + 0.11
    assert abs(DD0 - 1553.02734375) < 1e-12


def test_rel_round_trips_and_the_tick_centre():
    rng = np.random.default_rng(2)
    rel = rng.uniform(-0.5, 0.5, 12)
    assert np.allclose(C.rel_from_servo_deg(C.servo_deg_from_rel(rel)), rel)
    ticks = rng.integers(300, 700, 12)
    assert np.array_equal(C.ticks_from_servo_deg(C.servo_deg_of_tick(ticks)), ticks)
    assert np.array_equal(C.ticks_from_servo_deg(C.servo_deg_from_rel(C.rel_from_ticks(ticks))), ticks)
    dd, back = C.quantise_rel(rel)
    assert np.allclose(back, C.rel_from_dd(dd)) and np.abs(back - rel).max() < np.radians(0.5)


def test_tendon_coefficients_and_outward_signs():
    for robot in robots.ROBOTS:
        c = Contract.load(robot=robot)
        kc = contract_tendon_coefs(c)
        assert kc.shape == (4, 2) and np.all(kc[:, 1] != 0)
    assert leg_set_name(None) == leg_set_name("new") == "sku_2"
    with pytest.raises(ValueError):
        leg_set_name("other")


def test_sku1_calibration_file():
    d = json.loads(robots.SKU1.contract_json.read_text())
    assert d["joint_order"] == json.loads(robots.SKU2.contract_json.read_text())["joint_order"]
    assert len(d["stand_dd"]) == 12 and len(d["offsets_deg"]) == 12
