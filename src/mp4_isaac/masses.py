"""Weighed rigid-body masses, written onto the articulation after spawn.

The USD keeps the SolidWorks plastic. This writes ``sku2_masses.json`` (or
the 478 g heatsink file) onto the Isaac bodies and leaves ``pupper.xml``
unchanged. SKU1 has no mass file; its 360 g model is the ``<inertial>`` set
already in ``sku_1/pupper.xml``.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import torch

from .paths import asset_dir

LEGS = ("FR", "FL", "RR", "RL")
PREFIX = {"FR": "front_right", "FL": "front_left", "RR": "back_right", "RL": "back_left"}


def link_name(leg: str, part: str) -> str:
    """MuJoCo ``FR_thigh`` -> Isaac ``front_right_leg_upper_link``."""
    prefix = PREFIX[leg]
    if part == "hip":
        return f"{prefix}_hip_link"
    if part == "thigh":
        return f"{prefix}_leg_upper_link"
    if part == "calf":
        return f"{prefix}_leg_lower_link"
    raise ValueError(part)


def sku2_budget(heatsink: bool = False) -> dict:
    name = "sku2_masses_478g.json" if heatsink else "sku2_masses.json"
    return json.loads((asset_dir("sku_2") / name).read_text())


def sku2_link_masses(cad: dict[str, float], budget: dict) -> dict[str, float]:
    """``apply_masses``: each hip module, each leg's 6.5 g by the CAD thigh/calf ratio, base the rest."""
    assigned: dict[str, float] = {}
    for leg in LEGS:
        assigned[link_name(leg, "hip")] = float(budget["module_kg"])
        thigh = link_name(leg, "thigh")
        calf = link_name(leg, "calf")
        pair = cad[thigh] + cad[calf]
        if pair <= 0.0:
            raise ValueError(f"{leg} CAD thigh+calf mass is {pair}")
        scale = float(budget["leg_kg"]) / pair
        assigned[thigh] = cad[thigh] * scale
        assigned[calf] = cad[calf] * scale
    others = sum(mass for name, mass in cad.items() if name not in assigned and name != "base_link")
    assigned["base_link"] = float(budget["total_kg"]) - sum(assigned.values()) - others
    if assigned["base_link"] <= 0.0:
        raise ValueError(f"base mass came out {assigned['base_link']}")
    return assigned


def _fullinertia_matrix(values: list[float]) -> list[float]:
    """MuJoCo ``fullinertia`` (ixx iyy izz ixy ixz iyz) as a row-major 3x3 about the body origin's COM."""
    ixx, iyy, izz, ixy, ixz, iyz = values
    return [ixx, ixy, ixz, ixy, iyy, iyz, ixz, iyz, izz]


def sku1_inertials() -> dict[str, dict]:
    """Body mass, COM and inertia from ``sku_1/pupper.xml``, skipping the ghost ``g_*`` copies."""
    root = ET.parse(asset_dir("sku_1") / "pupper.xml").getroot()
    found: dict[str, dict] = {}
    for body in root.iter("body"):
        name = body.get("name") or ""
        if not name or name.startswith("g_"):
            continue
        inertial = next((child for child in body if child.tag == "inertial"), None)
        if inertial is None:
            continue
        if name == "base":
            link = "base_link"
        else:
            leg, part = name.split("_", 1)
            link = link_name(leg, part)
        pos = [float(v) for v in inertial.get("pos", "0 0 0").split()]
        mass = float(inertial.get("mass"))
        full = [float(v) for v in inertial.get("fullinertia").split()]
        found[link] = {"mass": mass, "com": pos, "inertia": _fullinertia_matrix(full)}
    expected = {"base_link"} | {link_name(leg, part) for leg in LEGS for part in ("hip", "thigh", "calf")}
    missing = expected - set(found)
    if missing:
        raise ValueError(f"sku_1/pupper.xml is missing inertials for {sorted(missing)}")
    return found


def _tensor(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def _index(names: list[str], link: str) -> int:
    if link not in names:
        raise KeyError(f"{link} is not an articulation body; bodies are {names}")
    return names.index(link)


def apply_masses(robot, sku: str, *, heatsink: bool = False) -> dict[str, float]:
    """Write the weighed masses. Returns the mass written on each named link."""
    names = list(robot.body_names)
    current = _tensor(robot.data.body_mass).detach().clone()
    inertia = _tensor(robot.data.body_inertia).detach().clone()
    com_pos = _tensor(robot.data.body_com_pos_b).detach().clone()
    com_quat = _tensor(robot.data.body_com_quat_b).detach().clone()
    cad = {name: float(current[0, i]) for i, name in enumerate(names)}

    if sku == "sku_2":
        budget = sku2_budget(heatsink)
        written = sku2_link_masses(cad, budget)
        masses = current.clone()
        for link, mass in written.items():
            index = _index(names, link)
            scale = mass / cad[link] if cad[link] > 0.0 else 1.0
            masses[:, index] = mass
            inertia[:, index] *= scale
        base = _index(names, "base_link")
        com_pos[:, base, 0] = float(budget["base_com_x_m"])
    elif sku == "sku_1":
        spec = sku1_inertials()
        written = {link: float(item["mass"]) for link, item in spec.items()}
        masses = current.clone()
        for link, item in spec.items():
            index = _index(names, link)
            masses[:, index] = item["mass"]
            inertia[:, index] = torch.tensor(item["inertia"], dtype=inertia.dtype, device=inertia.device)
            com_pos[:, index] = torch.tensor(item["com"], dtype=com_pos.dtype, device=com_pos.device)
            com_quat[:, index] = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=com_quat.dtype, device=com_quat.device)
    else:
        raise ValueError(f"no mass model for {sku}")

    robot.set_inertias_index(inertias=inertia)
    robot.set_masses_index(masses=masses)
    pose = torch.cat([com_pos, com_quat], dim=-1)
    robot.set_coms_index(coms=pose)
    total = float(masses[0].sum())
    label = "478 g heatsink" if sku == "sku_2" and heatsink else sku
    print(f"masses: {label}  total {total * 1000:.1f} g  base {written['base_link'] * 1000:.1f} g")
    return written
