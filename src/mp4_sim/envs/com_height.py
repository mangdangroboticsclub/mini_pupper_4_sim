"""The whole robot's centre-of-mass height, randomised per robot (``com_height_mm``).

The flag gives the WHOLE robot's centre-of-mass height in mm above the floor (the feet's contact points) at the
contract's stand, the base level (in the base frame z = height - ``stand_geometry``'s ``base_z_m``): the height a
weighing on scales with one end raised gives. The SKU1's masses file has 65.9 mm.

The base alone (about 120 g of SKU1's 413 g) cannot move the whole robot's centre of mass 16 mm (it would need about
55 mm, outside its 27 mm box), so one shared draw ``d`` per robot moves the centres of mass of the base AND the four
``{leg}_hip`` module bodies together, all by ``d`` along the base's z at the stand (``shift_com_height``). Not
``envs_mdp.dr.body_com_offset``: it offsets one local axis on every body, and the hip bodies' frames are turned (their
local y is the base's z at the stand; the axis and sign per body are read from the model when the range is made). The
whole robot's centre of mass moves by ``d * m_moved / M``, so the draw's range is ``(height - h0) * M / m_moved``, from
the plant's nominal masses (the options' masses file; ``h0`` the nominal whole-robot height). The event is installed
last, after the attachment's draw (the lidar's mass combined into the base): with the lidar on, the base body that moves
carries it too, so its share of the shift is the base's (the plant's other draws act as well: base mass x U(0.9, 1.1),
the base's own COM noise).
"""
from __future__ import annotations

import functools

import numpy as np

COM_BODIES = ("base", "FR_hip", "FL_hip", "RR_hip", "RL_hip")
EVENT = "com_height"


@functools.lru_cache(maxsize=None)
def nominal(masses_json, legs) -> dict:
    """The plant's nominal whole-robot centre of mass at the stand (``legs``: a ``robot.model.Legs``): height above
    the floor (m), total mass, the moved bodies' mass, and the check that the moved bodies' frames are the base's."""
    from mp4_sim.robot import model as robot_model
    m = robot_model.get_spec(masses_json=masses_json, legs=legs).compile()
    d = robot_model.stand_data(m, legs.robot)
    base = m.body("base").id
    base_z = robot_model.stand_geometry(legs)["base_z_m"]
    com = d.subtree_com[base] - d.xpos[base]
    ids = [m.body(n).id for n in COM_BODIES]
    rb = d.xmat[base].reshape(3, 3)
    axes, signs = [], []
    for i in ids:
        zrow = (rb.T @ d.xmat[i].reshape(3, 3))[2]      # the base's z in the body's local axes
        k = int(np.argmax(np.abs(zrow)))
        if abs(abs(zrow[k]) - 1.0) > 1e-9:
            raise ValueError(f"body {m.body(i).name}: no local axis along the base's z at the stand")
        axes.append(k)
        signs.append(float(np.sign(zrow[k])))
    return {"height_m": float(com[2] + base_z), "com_base_frame_m": [float(v) for v in com],
            "total_kg": float(m.body_mass.sum()), "moved_kg": float(m.body_mass[ids].sum()),
            "base_z_m": float(base_z), "axes": tuple(axes), "signs": tuple(signs)}


def body_range(com_height_mm, masses_json, legs) -> tuple:
    """The five bodies' shared z offset range (m) giving the whole robot ``com_height_mm`` (LO, HI) at nominal masses."""
    lo, hi = (float(v) for v in com_height_mm)
    if not 0.0 < lo <= hi < 200.0:
        raise ValueError(f"com_height_mm {com_height_mm}: LO <= HI, mm above the floor")
    n = nominal(masses_json, legs)
    k = n["total_kg"] / n["moved_kg"]
    return ((lo / 1000.0 - n["height_m"]) * k, (hi / 1000.0 - n["height_m"]) * k)


def shift_com_height(env, env_ids, dz_range: tuple, bodies: tuple, axes: tuple, signs: tuple) -> None:
    """Startup event: one draw ``d`` ~ U(dz_range) per robot (m), added to each body's centre of mass along its local
    axis ``axes[k]`` times ``signs[k]`` (the base's z). Keeps the draws in ``env.com_height_draw``."""
    import torch
    robot = env.scene["robot"]
    model = env.sim.model
    lo, hi = dz_range
    d = lo + (hi - lo) * torch.rand(env.num_envs, device=env.device)
    for name, ax, sg in zip(bodies, axes, signs):
        b = robot.indexing.body_ids[robot.find_bodies(name)[0][0]]
        model.body_ipos[:, b, ax] += sg * d
    env.com_height_draw = d


def _decorated():
    from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
    return requires_model_fields("body_ipos", recompute=RecomputeLevel.set_const)(shift_com_height)


def event_cfg(opt):
    """The startup event ``com_height`` for ``opt.com_height_mm`` (``opt``: options with ``com_height_mm``,
    ``masses_json`` and the legs' fields), or None when the option is off (the plant and its random stream as
    before). It comes last, after the attachment's draw."""
    rng = getattr(opt, "com_height_mm", None)
    if rng is None:
        return None
    from mjlab.managers.event_manager import EventTermCfg

    from mp4_sim.robot.model import legs_of
    n = nominal(opt.masses_json, legs_of(opt))
    r = body_range(rng, opt.masses_json, legs_of(opt))
    return EventTermCfg(func=_EVENT_FUNC(), mode="startup",
                        params={"dz_range": r, "bodies": COM_BODIES, "axes": n["axes"], "signs": n["signs"]})


def install_com_height(cfg, opt) -> None:
    """On an env config: call last (after the attachment): ``cfg.events["com_height"]`` = ``event_cfg(opt)`` when
    the option is on."""
    ev = event_cfg(opt)
    if ev is not None:
        cfg.events[EVENT] = ev


@functools.lru_cache(maxsize=None)
def _EVENT_FUNC():
    return _decorated()


def describe(opt) -> dict | None:
    """What a run's config records: the flag, the bodies, their range, the nominal height."""
    rng = getattr(opt, "com_height_mm", None)
    if rng is None:
        return None
    from mp4_sim.robot.model import legs_of
    n = nominal(opt.masses_json, legs_of(opt))
    r = body_range(rng, opt.masses_json, legs_of(opt))
    return {"com_height_mm": [float(v) for v in rng], "bodies": list(COM_BODIES), "body_dz_range_m": list(r),
            "local_axes": list(n["axes"]), "signs": list(n["signs"]),
            "nominal_height_mm": round(n["height_m"] * 1000.0, 3), "total_kg": n["total_kg"], "moved_kg": n["moved_kg"],
            "frame": "whole robot, mm above the floor at the contract's stand"}
