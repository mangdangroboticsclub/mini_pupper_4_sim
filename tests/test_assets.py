"""The packaged robot files: every model file, mesh and measured file at its recorded sha256, and the data files the
simulation reads."""
import json
import shutil
import subprocess

import mujoco
import pytest

from mp4_sim import paths, robots
from mp4_sim.robot import model

# Data whose hashes the policies trained on these models record, and the servo fits
INVARIANTS = {
    "robot/assets/sku_2/measured/sku2_contract.json": "86f1d330a9f30d7e2550d084e0593ef48d16ae4fc9970a9c32e2746f9cea7375",
    "robot/assets/sku_2/measured/sku2_joint_limits.json":
        "9cad8e76f4b8bf505c4283905106be9786ef2c81a2bffa75ece404c51c026312",
    "robot/assets/sku_2/measured/sku2_hip_envelope.json":
        "0d9d93b40598af6d0cc7feca381297077e9a500075d31de67f0848f3d9acac13",
    "robot/assets/sku_2/measured/sku2_masses.json": "823884dc6eb1ad710395357013cc0b7971e303b1fb9d8def4579861c1800c00b",
    "robot/assets/sku_2/measured/imu_mount.json": "0c1df51a25f121180858a5464ec15513c40e760552b1b15c9c80e80912845d87",
    "robot/assets/sku_2/pupper.xml": "455316472c9c059c7bb9e2d5236f87c59a6d58e42071c89507fcbe8bf83f3dcf",
    "robot/assets/sku_1/pupper.xml": "5ce84acc8ce53d1597d2f5df8c8ffc1ba3b56055e3f489c395cfc9211d2c2217",
    "robot/assets/sku_1/measured/sku1_contract.json": "48720bbdd8828779189aa5b1b2df7d0f01055870ea55639a31f8eaa1e9796bbe",
    "actuators/params/md300/md300_id_1_m6.json": "167db136411b2cd90f19967249394fbd72c9cb06650868b01f72f273730f0f27",
    "actuators/params/md300/md300_id_2_m6.json": "304edbc8551c682a5e398328be814f76e631506264c2d9bc9edf464810960037",
    "actuators/params/md300/md300_id_3_m6.json": "330cc38b64416421130ec73470aa055b87a2d4e9c66251e002f936b8f7498336",
    "actuators/params/md01/bam_t5ub06_id_2_m6.json": "d99fae0ed91bed9261d0ff6eb8362e76d91d3bfe150839cdf2d89cac8058a202",
    "actuators/params/md01/bam_t5ub06_id_3_m6.json": "16596b33c70a5fd574158292425b81b55ee988a11221ea9fae7089caf46fb601",
}


@pytest.mark.parametrize("rel", sorted(INVARIANTS))
def test_invariant_files(rel):
    assert paths.sha256(paths.PACKAGE / rel) == INVARIANTS[rel]


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_every_robot_file_matches_its_recorded_hash(robot):
    out = robots.check_files(robot)
    r = robots.get(robot)
    assert set(r.hashes) <= set(out)
    assert sum(1 for k in out if k.startswith("parts/")) == len(list((r.dir / "parts").iterdir()))


def test_a_changed_file_is_refused(tmp_path):
    import shutil
    from dataclasses import replace
    r = robots.SKU2
    d = tmp_path / "sku_2"
    shutil.copytree(r.dir, d)
    fake = replace(r, dir=d, hashes=dict(r.hashes))
    robots.check_files(fake)
    (d / "measured" / "sku2_masses.json").write_text((d / "measured" / "sku2_masses.json").read_text() + " ")
    with pytest.raises(ValueError, match="sku2_masses.json"):
        robots.check_files(fake)


def test_limits_files_carry_their_recorded_hashes():
    for r in (robots.SKU2, robots.SKU1):
        assert r.limits_sha256_file.read_text().split()[0] == paths.sha256(r.limits_json)
        assert r.limits_sha256_file.read_text().split()[1] == r.limits_json.name


@pytest.mark.parametrize("robot", robots.ROBOTS)
def test_every_mesh_the_model_names_is_packaged(robot):
    """SKU2's visual-only accessories are not packaged (``model.ACCESSORY_MESHES``); every other mesh is."""
    r = robots.get(robot)
    spec = mujoco.MjSpec.from_file(str(r.xml))
    for mesh in spec.meshes:
        packaged = (r.dir / "parts" / mesh.file).is_file()
        assert packaged == (robot == "sku1" or mesh.name not in model.ACCESSORY_MESHES), mesh.name


def test_stairs_and_tile_data():
    for name, steps in (("four_steps", 4), ("eight_steps", 8)):
        d = json.loads((paths.ENV_DATA / "stairs" / f"{name}.json").read_text())
        assert len(d["steps_mm"]) == steps and d["width_mm"] == 400
        assert all(r == 25 for r, _ in d["steps_mm"])
    assert (paths.ENV_DATA / "terrain_tile1_hmap.npz").is_file()


def test_paths_names_are_package_relative():
    p = robots.SKU2.masses_path
    assert paths.name(p) == "mp4_sim:robot/assets/sku_2/measured/sku2_masses.json"
    assert paths.resolve(paths.name(p)) == p
    assert paths.name("/elsewhere/x.json") == "/elsewhere/x.json"


def test_sha256_reads_crlf_text_as_lf(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b'{"x": 1}\n')
    b.write_bytes(b'{"x": 1}\r\n')
    assert paths.sha256(a) == paths.sha256(b)
    c, d = tmp_path / "c.stl", tmp_path / "d.stl"
    c.write_bytes(b"x\n")
    d.write_bytes(b"x\r\n")
    assert paths.sha256(c) != paths.sha256(d)


def test_text_files_hashed_as_bytes_keep_their_line_endings():
    """A text file hashed byte for byte (its suffix not in ``paths.TEXT_SUFFIXES``) is marked ``-text`` in
    ``.gitattributes``, so a checkout that converts line endings keeps its recorded hash."""
    repo = paths.PACKAGE.parents[1]
    if shutil.which("git") is None or not (repo / ".git").exists():
        pytest.skip("not a git checkout")
    files = []
    for r in (robots.SKU2, robots.SKU1):
        for p in list(r.files().values()) + [r.limits_sha256_file] + [
                r.dir / line.split(maxsplit=1)[1] for line in (r.dir / "SHA256SUMS").read_text().splitlines() if line]:
            if p.suffix.lower() not in paths.TEXT_SUFFIXES and b"\0" not in p.read_bytes():
                files.append(str(p.relative_to(repo)))
    assert any(f.endswith(".obj") for f in files) and any(f.endswith("SHA256SUMS") for f in files)
    out = subprocess.run(["git", "check-attr", "text", "--", *files], cwd=repo, capture_output=True, text=True,
                         check=True).stdout.splitlines()
    assert len(out) == len(files) and all(line.endswith(": text: unset") for line in out), out
