"""The measured joint limits (``limits``): the file and its recorded hash, the model ranges, applying them."""
import json
import math
import shutil

import numpy as np
import pytest

from mp4_sim import limits as L
from mp4_sim import paths, robots
from mp4_sim.calibration import ABD_OUT_SIGN, Contract
from mp4_sim.robot import model

C = Contract.load()


def test_hand_ranges_reproduce_the_files_model_ranges():
    """The mapping physical -> model (stand + radians(sign x value)) gives the file's own model_joint_range_rad."""
    for robot in robots.ROBOTS:
        hand = L.joint_ranges(hand_only=True, robot=robot)
        for leg in L.load(robot=robot)["legs"]:
            for name, r in leg["model_joint_range_rad"].items():
                if not name.endswith("_note"):
                    assert np.allclose(hand[name], r, atol=6e-5), (robot, name, hand[name], r)
        assert L.abd_out_sign(robot=robot) == ABD_OUT_SIGN


def test_ranges_take_the_narrower_of_hand_and_powered_per_side():
    j = L.load()
    phys = L.physical_ranges()
    for d in j["legs"]:
        assert d.get("abd_out_range_powered") is None
        assert phys[d["leg"]]["abd_out"] == tuple(d["abd_out_range"])
    hand = {d["leg"]: d["abd_out_range"] for d in j["legs"]}
    jr = L.joint_ranges()
    stand = dict(zip(C.joint_order, C.stand_joint_rad))
    fl, fr = hand["FL"], hand["FR"]
    assert np.allclose(np.degrees(np.subtract(jr["FL_hip"], stand["FL_hip"])), (-fl[1], -fl[0]))  # FL: outward is -
    assert np.allclose(np.degrees(np.subtract(jr["FR_hip"], stand["FR_hip"])), (fr[0], fr[1]))
    j2 = json.loads(json.dumps(j))
    d = next(x for x in j2["legs"] if x["leg"] == "FL")
    d["abd_out_range"], d["abd_out_range_powered"] = [-16.7, 25.9], [-13.8, 26.6]
    assert L.physical_ranges(j2)["FL"]["abd_out"] == (-13.8, 25.9)
    for name, (lo, hi) in jr.items():
        h = L.joint_ranges(hand_only=True)[name]
        assert h[0] <= lo + 1e-12 and hi <= h[1] + 1e-12


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_apply_limits_sets_the_ranges_on_the_spec(robot):
    legs = model.Legs(robot=robot)
    c = Contract.load(robot=robot)
    ranges = L.joint_ranges(robot=robot)
    spec = model.get_spec(legs=legs)
    got = L.apply_limits(spec, margin_deg=1.0, robot=robot)
    m = spec.compile()
    for name, (lo, hi) in ranges.items():
        j = m.joint(name)
        assert m.jnt_limited[j.id] == 1
        assert np.allclose(m.jnt_range[j.id], (lo + math.radians(1), hi - math.radians(1)))
        assert np.allclose(got[name], m.jnt_range[j.id])
    m0 = model.get_spec(legs=legs).compile()
    assert not any(m0.jnt_limited[m0.joint(n).id] for n in ranges)
    m1 = model.get_spec(limits=True, legs=legs).compile()
    for n, (lo, hi) in ranges.items():
        assert np.allclose(m1.jnt_range[m1.joint(n).id], (lo, hi))
    stand = dict(zip(c.joint_order, c.stand_joint_rad))
    for name, (lo, hi) in ranges.items():
        assert lo < stand[name] < hi
    with pytest.raises(ValueError):
        L.apply_limits(model.get_spec(legs=legs), margin_deg=90.0, robot=robot)


def test_load_refuses_a_file_that_differs_from_its_recorded_hash(tmp_path, monkeypatch):
    from dataclasses import replace
    d = tmp_path / "measured"
    d.mkdir()
    for p in (robots.SKU2.limits_json, robots.SKU2.limits_sha256_file):
        shutil.copy(p, d / p.name)
    fake = replace(robots.SKU2, limits_json=d / robots.SKU2.limits_json.name)
    assert L.load(robot=fake) == L.load()
    p = d / robots.SKU2.limits_json.name
    p.write_text(p.read_text().replace('"FR"', '"FR" ', 1))
    with pytest.raises(ValueError, match="recorded"):
        L.load(robot=fake)
    assert L.load(path=p)["legs"]                             # an explicit path is read unchecked
    assert L.recorded_sha256() == robots.SKU2.hashes["measured/sku2_joint_limits.json"]


def test_limits_were_traced_at_each_robots_calibration(tmp_path):
    for robot in robots.ROBOTS:
        rec = L.limits_contract_check(robot=robot)
        assert rec["match"] and rec["contract_in_force"].startswith("mp4_sim:robot/assets/")
    rec = L.limits_contract_check(allow_mismatch=True, robot="sku2", contract_json=robots.SKU1.contract_json)
    assert not rec["match"]
    with pytest.raises(ValueError):
        L.limits_contract_check(robot="sku2", contract_json=robots.SKU1.contract_json)
    # the calibration's hash is the LF one (paths.sha256): a CRLF copy matches
    crlf = tmp_path / "sku2_contract.json"
    crlf.write_bytes(robots.SKU2.contract_json.read_bytes().replace(b"\n", b"\r\n"))
    assert L.limits_contract_check(robot="sku2", contract_json=crlf)["match"]


def test_track_and_envelope():
    lo, hi = L.track_rel(C, 20.0)
    assert np.all(lo < 0) and np.all(hi > 0)
    lo2, hi2 = L.track_rel(C, -30.0)
    assert np.all(lo2 < lo) and np.all(hi2 > hi)
    assert np.allclose(L.track_rel(C), (lo, hi))
    env = L.hip_envelope()
    assert set(env) == {"FR", "FL", "RR", "RL"}
    assert L.hip_envelope_path().name == "sku2_hip_envelope.json"
    assert L.hip_envelope_path("sku1").name == "sku1_hip_envelope.json"
    assert L.hip_envelope(path=paths.PACKAGE / "nothing.json") is None
    assert L.thigh_stops_deg() == {} or all(v > 0 for v in L.thigh_stops_deg().values())


def test_sku1_limits_are_its_own_traced_limits_at_its_calibration():
    j1 = L.load(robot="sku1")
    assert "provisional" not in j1 and "hip_envelope_json" not in j1
    assert L.limits_contract_check(robot="sku1")["match"]
    p1, p2 = L.physical_ranges(robot="sku1"), L.physical_ranges(robot="sku2")
    assert p1 != p2
    for leg in j1["legs"]:
        assert leg["knee_fold_range"][1] == 49.0
    assert [leg["abd_out_range"] for leg in j1["legs"]] == [[-22.8, 25.1], [-18.0, 26.8], [-22.6, 30.3], [-22.1, 27.9]]
    assert robots.SKU1.provisional == ("IMU mount and noise profile: SKU2's",)
    env = L.hip_envelope(robot="sku1")
    assert set(env) == {"FR", "FL", "RR", "RL"}
