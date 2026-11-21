"""The built wheel carries the package data: the models, meshes, measured files, servo fits and terrain data."""
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "mp4_sim"
DATA_SUFFIXES = {".stl", ".STL", ".xml", ".json", ".npz", ".sha256", ".urdf", ".obj", ".png"}


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not found (it builds the wheel with hatchling)")
    out = tmp_path_factory.mktemp("wheel")
    r = subprocess.run([uv, "build", "--wheel", "--offline", "--out-dir", str(out), str(ROOT)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        if "offline" in (r.stderr + r.stdout).lower():
            pytest.skip("hatchling is not in uv's cache and the build is offline")
        raise AssertionError(r.stderr[-2000:])
    (whl,) = out.glob("mp4_sim-*.whl")
    return zipfile.ZipFile(whl)


def test_wheel_has_every_data_file(wheel):
    names = set(wheel.namelist())
    want = {"mp4_sim/" + p.relative_to(PKG).as_posix() for p in PKG.rglob("*")
            if p.is_file() and (p.suffix in DATA_SUFFIXES or p.name in ("SHA256SUMS", "LICENSE"))
            and "__pycache__" not in p.parts}
    assert want <= names, sorted(want - names)[:10]
    for must in ("mp4_sim/robot/assets/sku_2/pupper.xml", "mp4_sim/robot/assets/sku_2/parts/FR_foot.stl",
                 "mp4_sim/robot/assets/sku_1/parts/base_screen_face.obj",
                 "mp4_sim/robot/assets/sku_2/measured/sku2_joint_limits.sha256",
                 "mp4_sim/actuators/params/md300/md300_id_1_m6.json", "mp4_sim/envs/data/terrain_tile1_hmap.npz",
                 "mp4_sim/envs/data/stairs/four_steps.json", "mp4_sim/actuators/bam/LICENSE"):
        assert must in names, must
    counts = {s: sum(1 for n in names if n.endswith(s)) for s in (".stl", ".xml", ".json", ".npz")}
    assert counts[".stl"] >= 75 and counts[".xml"] >= 4 and counts[".json"] >= 18 and counts[".npz"] == 1


def test_wheel_version_and_no_tests(wheel):
    names = wheel.namelist()
    assert any(n.startswith("mp4_sim-0.2.0.dist-info/") for n in names)
    assert not any(n.startswith("tests/") for n in names)
