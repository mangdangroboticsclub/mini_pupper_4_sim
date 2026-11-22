import json

import mujoco
import numpy as np
import pytest

from mp4_sim import paths, robots
from mp4_sim.calibration import ACTION_SIZE, Contract
from mp4_sim.robot import model
from mp4_sim.robot.mapping import ServoMap


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_contract_loads(robot):
    c = Contract.load(robot=robot)
    assert len(c.signs) == ACTION_SIZE
    assert len(c.joint_order) == ACTION_SIZE
    assert set(np.abs(c.signs)) == {1.0}


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_spec_applies_weighed_masses(robot):
    r = robots.get(robot)
    m = model.get_spec(legs=model.Legs(robot=robot)).compile()
    masses = json.loads(r.masses_path.read_text())
    assert m.body_mass.sum() == pytest.approx(masses["total_kg"])
    assert m.body("FR_hip").mass[0] == pytest.approx(masses["module_kg"])
    assert m.nu == 0
    imu = {"imu_ang_vel": mujoco.mjtSensor.mjSENS_GYRO,
           "imu_lin_vel": mujoco.mjtSensor.mjSENS_VELOCIMETER,
           "imu_lin_acc": mujoco.mjtSensor.mjSENS_ACCELEROMETER}
    for name, kind in imu.items():
        assert m.sensor(name).type[0] == kind


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_model_files_match_the_registry(robot):
    r = robots.get(robot)
    assert paths.sha256(r.xml) == r.hashes["pupper.xml"]
    assert paths.sha256(r.servo_map_json) == r.hashes["servo_map.json"]


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
    with pytest.raises(ValueError):
        arms.bam("scaled", arms.T5UB06, scales=(("armature", 0.5),))
