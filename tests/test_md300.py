"""md300: the vendored MD-300-150 model and its gains (``actuators.choice``, ``servo_model``), on the CPU."""
import json
import math
from argparse import ArgumentParser

import numpy as np
import pytest

from mp4_sim import paths
from mp4_sim.actuators import choice
from mp4_sim.actuators.bam.model import load_model

SETS = {1: "md300_id_1_m6.json", 2: "md300_id_2_m6.json", 3: "md300_id_3_m6.json"}
MD300 = paths.ACTUATOR_PARAMS / "md300"


@pytest.mark.parametrize("k", sorted(SETS))
def test_parameter_files_are_the_published_m6(k):
    d = json.loads((MD300 / SETS[k]).read_text())
    assert d["model"] == "m6" and d["actuator"] == f"mangdang_md300_id_{k}"


def ours(k, vin=12.0):
    m = load_model(str(MD300 / SETS[k]))
    act = m.actuator
    assert type(act).__name__ == "MD300Actuator"
    choice.apply_md300_firmware(act, {**choice.MD300_FIRMWARE, "vin": vin})
    return m


def test_firmware_law_by_hand():
    """duty = pwm_scale kp (err_deg - 0.1) - kd 2.87e-5 (pwm_scale/0.001) kd_ratio dq, x error_gain_ratio on the P."""
    m = ours(2)
    act = m.actuator
    egr, kdr = m.error_gain_ratio.value, m.kd_ratio.value
    err_deg, dq = 3.0, 0.7
    duty = 0.001 * 80.0 * (err_deg - 0.1) * egr - 400.0 * 2.87e-5 * kdr * dq
    ctl = act.compute_control(math.radians(err_deg), 0.0, dq, 0.001)
    assert ctl == pytest.approx(12.0 * duty, rel=1e-9)
    assert act.compute_control(math.radians(90.0), 0.0, 0.0, 0.001) == pytest.approx(12.0 * 0.99)


def test_plant_law_and_dead_band():
    for k in SETS:
        for vin in (12.0, 10.5):
            a = ours(k, vin)
            p = {n: getattr(a, n).value for n in ("kt", "kt_emf", "R")}
            q0 = 0.3
            for err, dq in ((math.radians(0.05), 0.0), (math.radians(2.0), 1.5), (math.radians(-60.0), 6.0)):
                ca = a.actuator.compute_control(q0 + err, q0, dq, 0.001)
                ta = a.actuator.compute_torque(ca, True, q0, dq)
                assert ta == pytest.approx(p["kt"] * (ca - p["kt_emf"] * dq) / p["R"], rel=1e-12, abs=1e-15)
                if abs(err) <= math.radians(0.1) and dq == 0.0:
                    assert abs(ca) < 1e-9
                assert abs(ca / vin) <= choice.ROBOT_MAX_PWM + 1e-12


def test_options_and_flags():
    ap = ArgumentParser()
    choice.add_args(ap, evaluation=True)
    a = ap.parse_args([])
    assert a.actuator is None and choice.options_from_args(a) == {
        "supply_v_range": None, "supply_drop_ohm_range": None, "r_scale_range": None}
    a = ap.parse_args(["--actuator", "md300", "--supply-v", "10.5", "12.6", "--r-scale", "1.0", "1.4"])
    assert choice.options_from_args(a) == {"actuator": "md300", "supply_v_range": (10.5, 12.6),
                                           "supply_drop_ohm_range": None, "r_scale_range": (1.0, 1.4)}
    with pytest.raises(ValueError):
        choice.check_name("md02")
    with pytest.raises(ValueError):
        choice._range((12.6, 10.5))
    assert choice.duty_per_ma("md01") == choice.duty_per_ma("md300") == 0.001


def test_dead_band_and_saturation_numpy_arrays():
    act = ours(3).actuator
    err = np.radians(np.array([0.0, 0.05, 0.2, 100.0]))
    ctl = act.compute_control(err, np.zeros(4), np.zeros(4), 0.001)
    assert ctl[0] == 0.0 and ctl[1] == 0.0 and ctl[2] > 0.0 and ctl[3] == pytest.approx(12.0 * 0.99)


def test_servo_model_sets_and_records():
    pytest.importorskip("mjlab")
    from mp4_sim.actuators import servo_model
    m = servo_model.md300()
    assert m.name == "md300" and len(m.actuators) == 3
    roles = m.params["roles"]
    assert [roles[r]["params"] for r in ("abd", "hip", "knee")] == [
        f"mp4_sim:actuators/params/md300/md300_id_{k}_m6.json" for k in (1, 2, 3)]
    assert all(roles[r]["sha256"] == paths.sha256(paths.resolve(roles[r]["params"])) for r in roles)
    assert servo_model.md01().latency_steps == (1, 5, 1)
    r = servo_model.build("md300", (10.5, 12.6), (0.0, 0.3), (1.0, 1.4))
    assert r.params["randomise"] == {"supply_v_range": (10.5, 12.6), "supply_drop_ohm_range": (0.0, 0.3),
                                     "r_scale_range": (1.0, 1.4)}
    with pytest.raises(ValueError):
        servo_model.build("md300", supply_v_range=(0.0, 1.0))
