"""What each module needs: ``mp4_sim`` and ``mp4_sim.calibration`` run on numpy alone (a robot copies those two
files); the robot model and limits on MuJoCo without mjlab."""
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"


def _modules_after(code: str) -> set:
    out = subprocess.run([sys.executable, "-c", code + "\nimport sys; print(' '.join(sorted(sys.modules)))"],
                         capture_output=True, text=True, check=True, env={"PYTHONPATH": str(SRC), "PATH": ""})
    return set(out.stdout.split())


def test_package_init_imports_nothing():
    mods = _modules_after("import mp4_sim")
    assert not {m for m in mods if m.startswith("mp4_sim.")}
    assert "numpy" not in mods


def test_calibration_needs_numpy_only():
    mods = _modules_after("import mp4_sim.calibration")
    assert {m for m in mods if m.startswith("mp4_sim")} == {"mp4_sim", "mp4_sim.calibration"}
    assert not {"mujoco", "torch", "mjlab", "scipy"} & mods


def test_calibration_runs_as_a_copied_file(tmp_path):
    """The two files a robot copies, alone in a folder, with a calibration file passed by path."""
    pkg = tmp_path / "mp4_sim"
    pkg.mkdir()
    for f in ("__init__.py", "calibration.py"):
        (pkg / f).write_bytes((SRC / "mp4_sim" / f).read_bytes())
    cal = SRC / "mp4_sim" / "robot" / "assets" / "sku_2" / "measured" / "sku2_contract.json"
    code = (f"from mp4_sim.calibration import Contract; c = Contract.load({str(cal)!r}); "
            "print(c.stand_dd.tolist() == c.dd_from_ticks(c.ticks_from_servo_deg([0.0] * 12)).tolist())")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(tmp_path), "PATH": ""}, cwd=tmp_path)
    assert out.stdout.strip() == "True"


def test_model_and_limits_without_mjlab():
    mods = _modules_after("import mp4_sim.robot.model, mp4_sim.limits, mp4_sim.robots, mp4_sim.actuators.bank")
    assert "mjlab" not in mods and "torch" not in mods


@pytest.mark.parametrize("name", ["mp4_sim.envs.scene", "mp4_sim.envs.terrain_spawn", "mp4_sim.actuators.servo_model",
                                  "mp4_sim.actuators.arms", "mp4_sim.envs.patch_warp", "mp4_sim.envs.noise"])
def test_simulation_modules_import(name):
    pytest.importorskip("mjlab")
    __import__(name)
