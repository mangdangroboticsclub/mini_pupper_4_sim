import json

import mujoco
import numpy as np
import pytest

from mp4_sim import paths
from mp4_sim.contract import ACTION_SIZE, Contract
from mp4_sim.robot import sku2
from mp4_sim.robot.mapping import ServoMap


def test_contract_loads():
    c = Contract.load()
    assert len(c.signs) == ACTION_SIZE
    assert len(c.joint_order) == ACTION_SIZE
    assert set(np.abs(c.signs)) == {1.0}


def test_sku2_spec_applies_weighed_masses():
    model = sku2.get_spec().compile()
    masses = json.loads(paths.MASSES_JSON.read_text())
    assert model.body_mass.sum() == pytest.approx(masses["total_kg"])
    assert model.body("FR_hip").mass[0] == pytest.approx(masses["module_kg"])
    assert model.nu == 0
    imu = {"imu_ang_vel": mujoco.mjtSensor.mjSENS_GYRO,
           "imu_lin_vel": mujoco.mjtSensor.mjSENS_VELOCIMETER,
           "imu_lin_acc": mujoco.mjtSensor.mjSENS_ACCELEROMETER}
    for name, kind in imu.items():
        assert model.sensor(name).type[0] == kind


def test_sku2_contract_matches_assets():
    recorded = json.loads(paths.CONTRACT_JSON.read_text())["sources"]
    by_name = {k.rsplit("/", 1)[-1]: v for k, v in recorded.items()}
    assert by_name["pupper.xml"] == paths.sha256(paths.SKU2_XML)
    assert by_name["servo_map.json"] == paths.sha256(paths.SKU2_DIR / "servo_map.json")


@pytest.mark.parametrize("sku", ["sku_1", "sku_2"])
def test_servo_map_stand_is_zero_servo(sku):
    q = ServoMap(sku=sku).servo_to_joint(np.zeros(12))
    assert q.shape == (12,)
    assert np.all(np.isfinite(q))


def test_t5ub06_arm():
    pytest.importorskip("mjlab")
    from mp4_sim.actuators import arms

    arm = arms.get("bam_t5ub06")
    assert arm.latency_steps == (1, 5, 1)
    assert [r["params"].rsplit("/", 1)[-1] for r in arm.params["roles"].values()] == [
        "bam_t5ub06_id_3_m6.json", "bam_t5ub06_id_2_m6.json", "bam_t5ub06_id_3_m6.json"]
