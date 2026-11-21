"""The robot registry (``robots``): the robots, their files, ``--robot``, no process-wide selection."""
import argparse

import pytest

from mp4_sim import robots
from mp4_sim.robot import model


def test_registry():
    assert robots.ROBOTS == ("sku2", "sku1") and robots.DEFAULT == "sku2"
    assert robots.get() is robots.SKU2 and robots.get("sku1") is robots.SKU1 and robots.get(robots.SKU1) is robots.SKU1
    with pytest.raises(ValueError):
        robots.get("sku3")
    for r in (robots.SKU1, robots.SKU2):
        for p in (r.xml, r.scene_xml, r.servo_map_json, r.contract_json, r.limits_json, r.limits_sha256_file,
                  r.hip_envelope_json, r.imu_mount_json, r.noise_json, r.masses_path):
            assert p.is_file(), p
    assert robots.SKU2.top_z_m == 0.0265 and robots.SKU1.top_z_m == 0.0135
    assert robots.SKU1.imu_mount_json == robots.SKU2.imu_mount_json        # the SKU2's (listed in provisional)
    assert robots.SKU2.to_meta()["contract_json"] == "mp4_sim:robot/assets/sku_2/measured/sku2_contract.json"


def test_add_args():
    ap = argparse.ArgumentParser()
    robots.add_args(ap)
    assert ap.parse_args([]).robot is None
    assert robots.get(ap.parse_args([]).robot) is robots.SKU2
    assert ap.parse_args(["--robot", "sku1"]).robot == "sku1"
    with pytest.raises(SystemExit):
        ap.parse_args(["--robot", "sku3"])


def test_masses_names():
    assert robots.SKU2.masses_for("sku2_masses.json") == "sku2_masses.json"
    assert robots.SKU2.masses_for(None) is None
    assert robots.SKU1.masses_for(None) == "sku1_masses_413g.json"
    with pytest.warns(UserWarning, match="SKU2 default"):
        assert robots.SKU1.masses_for("sku2_masses.json") == "sku1_masses_413g.json"
    assert robots.SKU1.masses_for("/x/other.json") == "/x/other.json"
    assert model.masses_path(None, "sku1") == robots.SKU1.masses_path
    assert model.masses_path("sku2_masses.json", "sku2") == robots.SKU2.masses_path


def test_building_one_robot_does_not_change_the_other():
    """Nothing is selected process-wide: building SKU1 leaves every SKU2 default as it was."""
    before = (model.legs_meta(), model.init_base_z(), model.Legs().contract().stand_dd.tolist())
    model.get_spec(legs=model.Legs(robot="sku1")).compile()
    model.legs_meta(model.Legs(robot="sku1"))
    after = (model.legs_meta(), model.init_base_z(), model.Legs().contract().stand_dd.tolist())
    assert before == after
