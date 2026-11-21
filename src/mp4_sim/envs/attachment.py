"""An attachment on the robot's hood: one collision cylinder of the body's collision class, and its mass on the base.

The cylinder is as long as the tallest height drawn. A robot's height is how far the cylinder stands out of the hood: the
rest is sunk into the body, and height 0 is fully inside it (with its centre past x = 52.5 mm a radius of 18.5 mm reaches
past the hood's front, over the lower box, by up to 2.5 mm at 55 mm).

Height, mass, x (and y, ``y_mm``, when it is a range) are drawn per robot once, at startup, after the plant's other
draws; y is drawn last, so a fixed y (the default 0) leaves the random stream as it was. ``present`` (a share, e.g. 0.5):
each robot then carries the attachment with that probability and nothing otherwise (height 0, mass 0: the lidar present
or absent per robot); None (the default) draws no presence, so the random stream is the one before the option. The mass sits at the centre of the
part standing out, as a solid cylinder of that height, and combines with the robot's own drawn mass, center of mass and
inertia. The base's center of mass moves, and its inertia tensor is recomposed about it (parallel axes, the cylinder's own
inertia) and diagonalised again, so the inertia's principal axes turn per robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch
from mjlab.managers.event_manager import EventTermCfg, RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import matrix_from_quat, quat_from_matrix

HOOD_TOP_Z = 0.0265     # m: the top of the base's upper collision box, in the base frame (SKU2; a robot's own:
                        # ``mp4_sim.robots.Robot.top_z_m``, ``install_attachment``'s ``top_z``)
GEOM = "attachment"
CONTACT_FIELDS = ("contype", "conaffinity", "priority", "condim", "solmix", "solref", "solimp", "friction", "margin",
                  "gap")


@dataclass(frozen=True)
class Attachment:
    height_mm: tuple = (0.0, 40.0)      # standing out of the hood
    mass_g: tuple = (0.0, 70.0)
    x_mm: tuple = (35.0, 55.0)          # the cylinder's axis, forward of the base origin
    radius_mm: float = 18.5
    present: float | None = None        # the share of robots carrying it (None: every robot)
    y_mm: tuple = (0.0, 0.0)            # the cylinder's axis, left of the base's mid-plane

    def __post_init__(self):
        for lo, hi in (self.height_mm, self.mass_g):
            if not 0.0 <= lo <= hi:
                raise ValueError(f"not a range: {self}")
        for lo, hi in (self.x_mm, self.y_mm):
            if not lo <= hi:
                raise ValueError(f"not a range: {self}")
        if self.radius_mm <= 0.0:
            raise ValueError(f"radius: {self}")
        if self.present is not None and not 0.0 <= self.present <= 1.0:
            raise ValueError(f"present: a share in [0, 1]: {self}")

    @classmethod
    def parse(cls, text: str) -> "Attachment":
        """One fixed attachment from ``height_mm,radius_mm,x_m,mass_g``."""
        height, radius, x, mass = (float(v) for v in text.split(","))
        return cls((height, height), (mass, mass), (x * 1000.0, x * 1000.0), radius)


def add_cylinder(spec: mujoco.MjSpec, att: Attachment, top_z: float = HOOD_TOP_Z) -> None:
    """The cylinder on ``base`` with the contact parameters of the base's collision boxes; no geom if it never stands out.
    ``top_z``: the top of the body it stands on (base frame, m)."""
    tallest = att.height_mm[1] / 1000.0
    if tallest == 0.0:
        return
    base = spec.body("base")
    body = next(g for g in base.geoms if g.contype or g.conaffinity)
    g = base.add_geom()
    g.name = GEOM
    g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
    g.size = [att.radius_mm / 1000.0, tallest / 2.0, 0.0]
    g.pos = [sum(att.x_mm) / 2000.0, sum(att.y_mm) / 2000.0, top_z + tallest / 2.0]
    g.density = 0.0
    g.rgba = [0.1, 0.1, 0.1, 1.0]
    for f in CONTACT_FIELDS:
        setattr(g, f, getattr(body, f))


def combined_inertia(mass0, com0, inertia0, mass1, com1, inertia1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two rigidly joined masses: (mass, center of mass, inertia tensor about it). Each (n,), (n, 3), (n, 3, 3)."""
    def about(mass, d):
        eye = torch.eye(3, device=d.device, dtype=d.dtype)
        dd = (d * d).sum(-1)[:, None, None] * eye - d[:, :, None] * d[:, None, :]
        return mass[:, None, None] * dd

    mass = mass0 + mass1
    com = (mass0[:, None] * com0 + mass1[:, None] * com1) / mass[:, None]
    return mass, com, inertia0 + about(mass0, com0 - com) + inertia1 + about(mass1, com1 - com)


def principal(inertia: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(diagonal, quaternion of the principal frame) of inertia tensors (n, 3, 3), as MuJoCo's ``body_inertia`` and
    ``body_iquat``."""
    diagonal, axes = torch.linalg.eigh(inertia)
    axes = axes.clone()
    axes[:, :, 2] *= torch.linalg.det(axes).sign()[:, None]
    return diagonal, quat_from_matrix(axes)


def _draw(env, limits: tuple, unit: float) -> torch.Tensor:
    """A fixed value uses no random numbers: the same seed with and without it meets the same plant."""
    lo, hi = limits
    if lo == hi:
        return torch.full((env.num_envs,), lo * unit, device=env.device)
    return (lo + (hi - lo) * torch.rand(env.num_envs, device=env.device)) * unit


@requires_model_fields("geom_pos", "body_mass", "body_ipos", "body_inertia", "body_iquat",
                       recompute=RecomputeLevel.set_const)
def draw_attachment(env, env_ids, att: Attachment, top_z: float = HOOD_TOP_Z) -> None:
    """Startup event. Keeps the draws in ``env.attachment_draw`` (m, kg)."""
    robot = env.scene["robot"]
    model = env.sim.model
    height, mass, x = _draw(env, att.height_mm, 1e-3), _draw(env, att.mass_g, 1e-3), _draw(env, att.x_mm, 1e-3)
    if att.present is not None:
        on = torch.rand(env.num_envs, device=env.device) < att.present
        height, mass = torch.where(on, height, 0.0), torch.where(on, mass, 0.0)
    y = _draw(env, att.y_mm, 1e-3)
    env.attachment_draw = {"height_m": height, "mass_kg": mass, "x_m": x, "y_m": y}
    if att.height_mm[1] > 0.0:
        geom = robot.indexing.geom_ids[robot.find_geoms(GEOM)[0][0]]
        model.geom_pos[:, geom, 0] = x
        if att.y_mm != (0.0, 0.0):
            model.geom_pos[:, geom, 1] = y
        model.geom_pos[:, geom, 2] = top_z + height - att.height_mm[1] / 2000.0

    body = robot.indexing.body_ids[robot.find_bodies("base")[0][0]]
    rot = matrix_from_quat(model.body_iquat[:, body])
    inertia0 = rot @ torch.diag_embed(model.body_inertia[:, body]) @ rot.transpose(1, 2)
    centre = torch.stack([x, y, top_z + height / 2.0], -1)
    r2 = (att.radius_mm / 1000.0) ** 2
    across = mass * (3.0 * r2 + height ** 2) / 12.0
    inertia1 = torch.diag_embed(torch.stack([across, across, mass * r2 / 2.0], -1))
    total, com, tensor = combined_inertia(model.body_mass[:, body], model.body_ipos[:, body], inertia0,
                                          mass, centre, inertia1)
    diagonal, quat = principal(tensor)
    model.body_mass[:, body] = total
    model.body_ipos[:, body] = com
    model.body_inertia[:, body] = diagonal
    model.body_iquat[:, body] = quat


def wrap_spec_fn(prev, att: Attachment, top_z: float | None = None):
    """The robot's ``spec_fn`` with the cylinder added last (it copies the base's collision class as the spec has it)."""
    top = HOOD_TOP_Z if top_z is None else float(top_z)

    def spec_fn():
        spec = prev()
        add_cylinder(spec, att, top)
        return spec

    return spec_fn


def event_cfg(att: Attachment, top_z: float | None = None) -> EventTermCfg:
    """The startup event drawing each robot's attachment; it builds on the plant's draws, so it comes after them."""
    top = HOOD_TOP_Z if top_z is None else float(top_z)
    params = {"att": att} if top == HOOD_TOP_Z else {"att": att, "top_z": top}
    return EventTermCfg(func=draw_attachment, mode="startup", params=params)


def install_attachment(cfg, att: Attachment, top_z: float | None = None) -> None:
    """On an env config (``cfg.scene.entities["robot"]``, ``cfg.events``): call last, after every other spec edit and
    startup event. ``top_z``: the top of the robot's body (base frame, m; None: SKU2's ``HOOD_TOP_Z``)."""
    robot = cfg.scene.entities["robot"]
    robot.spec_fn = wrap_spec_fn(robot.spec_fn, att, top_z)
    cfg.events["attachment"] = event_cfg(att, top_z)
