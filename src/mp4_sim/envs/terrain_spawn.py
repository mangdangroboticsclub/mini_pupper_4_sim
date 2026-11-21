"""Fallen starts on terrain: the ground's height as a tensor, and where each robot spawns on a tile.

``TerrainSpawn`` is built by a task's reset event (the get-up's fallen starts) and kept as ``env.terrain_spawn``. Robots on a flat column spawn at
the tile's origin. On a rough family they spawn anywhere on the tile but the flat platform at its origin, on printed stairs
anywhere on the flight, above the highest ground under their footprint.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np
import torch

from mp4_sim.envs import terrain as terrain_mod

GRID_PITCH = 0.02       # m
FOOTPRINT_R = 0.11      # m: ground height at a spawn = the highest of a cross and its corners at this reach
TILE_EDGE = 0.25        # m: spawns stay this far from a tile's edge
PLATFORM_CLEAR = 0.22   # m: ... and this far from its origin, in x or y, so the robot is not on the spawn platform
STAIRS_MARGIN = 0.08    # m: printed stairs: spawns from this far before the first riser to this far past the last
STAIRS_SIDE = 0.04      # m: ... and this far inside the sides
UP_TILT_STAIRS_DEG, UP_TILT_DEG = 40.0, 25.0   # a flight slopes 26.6 deg and the walker tilts to 36 on one
UP_RMS_DEG, UP_GYRO, UP_FEET = 25.0, 1.5, 3


def terrain_geoms(m: mujoco.MjModel) -> list[int]:
    return [i for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i])
            and not m.body(int(m.geom_bodyid[i])).name.startswith("robot/")]


def height_grid(m: mujoco.MjModel, pitch: float = GRID_PITCH) -> tuple[np.ndarray, np.ndarray]:
    """The terrain's top surface on a grid read from the model's boxes and heightfields: (heights (nx, ny), the (x, y) of
    node (0, 0)). Nodes with no geom above them are 0."""
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    geoms = terrain_geoms(m)
    reach = []
    for g in geoms:
        pos, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            half = m.hfield_size[m.geom_dataid[g]][:2]
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            assert abs(R[2, 2] - 1.0) < 1e-9, "a terrain box tilted out of the horizontal"
            half = np.abs(R[:2, :2]) @ m.geom_size[g][:2]
        else:
            raise ValueError(f"terrain geom type {m.geom_type[g]}")
        reach.append(np.r_[pos[:2] - half, pos[:2] + half])
    reach = np.array(reach)
    x0, y0 = reach[:, :2].min(0)
    nx, ny = (np.ceil((reach[:, 2:].max(0) - (x0, y0)) / pitch).astype(int) + 1).tolist()
    h = np.full((nx, ny), -np.inf)
    for g, (xa, ya, xb, yb) in zip(geoms, reach):
        i0, i1 = max(0, math.ceil((xa - x0) / pitch - 1e-9)), min(nx - 1, math.floor((xb - x0) / pitch + 1e-9))
        j0, j1 = max(0, math.ceil((ya - y0) / pitch - 1e-9)), min(ny - 1, math.floor((yb - y0) / pitch + 1e-9))
        if i1 < i0 or j1 < j0:
            continue
        X, Y = np.meshgrid(x0 + np.arange(i0, i1 + 1) * pitch, y0 + np.arange(j0, j1 + 1) * pitch, indexing="ij")
        pos, R = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
            top = pos[2] + _hfield_at(m, int(m.geom_dataid[g]), X - pos[0], Y - pos[1])
        else:
            lx = (X - pos[0]) * R[0, 0] + (Y - pos[1]) * R[1, 0]
            ly = (X - pos[0]) * R[0, 1] + (Y - pos[1]) * R[1, 1]
            size = m.geom_size[g]
            inside = (np.abs(lx) <= size[0] + 1e-9) & (np.abs(ly) <= size[1] + 1e-9)
            top = np.where(inside, pos[2] + size[2], -np.inf)
        h[i0:i1 + 1, j0:j1 + 1] = np.maximum(h[i0:i1 + 1, j0:j1 + 1], top)
    h[~np.isfinite(h)] = 0.0
    return h, np.array([x0, y0])


def _hfield_at(m: mujoco.MjModel, hid: int, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A heightfield's elevation (m, bilinear) at (x, y) from its centre; rows run along y, columns along x."""
    nr, nc = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    data = m.hfield_data[m.hfield_adr[hid]:m.hfield_adr[hid] + nr * nc].reshape(nr, nc)
    rx, ry, top = m.hfield_size[hid][:3]
    u = np.clip((x + rx) / (2 * rx) * (nc - 1), 0.0, nc - 1.0)
    v = np.clip((y + ry) / (2 * ry) * (nr - 1), 0.0, nr - 1.0)
    i, j = np.minimum(u.astype(int), nc - 2), np.minimum(v.astype(int), nr - 2)
    fu, fv = u - i, v - j
    return top * ((1 - fv) * ((1 - fu) * data[j, i] + fu * data[j, i + 1])
                  + fv * ((1 - fu) * data[j + 1, i] + fu * data[j + 1, i + 1]))


class GroundGrid:
    """The terrain's height at any (x, y): bilinear on ``height_grid``, clamped at the grid's edge."""

    def __init__(self, m: mujoco.MjModel, device, pitch: float = GRID_PITCH, dtype=torch.float32):
        h, origin = height_grid(m, pitch)
        self.pitch = pitch
        self.heights = torch.as_tensor(h, dtype=dtype, device=device)
        self.origin = torch.as_tensor(origin, dtype=dtype, device=device)
        self._last = torch.tensor(h.shape, dtype=dtype, device=device) - 1.0
        r = FOOTPRINT_R
        self._cross = torch.tensor([[dx, dy] for dx in (-r, 0.0, r) for dy in (-r, 0.0, r)], dtype=dtype, device=device)

    def under(self, xy: torch.Tensor) -> torch.Tensor:
        u = ((xy - self.origin) / self.pitch).clamp(min=torch.zeros_like(self._last), max=self._last)
        i = u.long().clamp(min=torch.zeros(2, dtype=torch.long, device=u.device), max=self._last.long() - 1)
        f = u - i
        ix, iy = i[..., 0], i[..., 1]
        fx, fy = f[..., 0], f[..., 1]
        h = self.heights
        return ((1 - fx) * ((1 - fy) * h[ix, iy] + fy * h[ix, iy + 1])
                + fx * ((1 - fy) * h[ix + 1, iy] + fy * h[ix + 1, iy + 1]))

    def footprint(self, xy: torch.Tensor) -> torch.Tensor:
        return self.under(xy[..., None, :] + self._cross).amax(-1)


class TerrainSpawn:
    """Per env, the cell it is on and the spawn origin in ``env.scene.env_origins``. ``place`` draws a new one at every
    reset, a uniform cell unless ``fix_cells`` set them. ``flat`` / ``on_stairs`` (n,) say which up rule the robot is under."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.terrain = env.scene.terrain
        subs = self.terrain.cfg.terrain_generator.sub_terrains
        self.columns = list(subs)
        self.families = [terrain_mod.family_of(c) for c in self.columns]
        self.ground = GroundGrid(env.sim.mj_model, dev)
        self.flat_col = torch.tensor([f == "flat" for f in self.families], device=dev)
        self.stairs_col = torch.tensor([f.startswith("stairs") for f in self.families], device=dev)
        self.region = torch.tensor([_region(subs[c], terrain_mod.family_of(c)) for c in self.columns],
                                   dtype=torch.float32, device=dev)     # (cols, 5): x0, x1, y0, y1, clear
        self.rows, self.cols = self.terrain.terrain_origins.shape[:2]
        self.fixed: tuple[torch.Tensor, torch.Tensor] | None = None
        self.row = torch.zeros(n, dtype=torch.long, device=dev)
        self.col = torch.zeros(n, dtype=torch.long, device=dev)
        self.flat = torch.zeros(n, dtype=torch.bool, device=dev)
        self.on_stairs = torch.zeros(n, dtype=torch.bool, device=dev)

    def fix_cells(self, rows: torch.Tensor, cols: torch.Tensor) -> None:
        self.fixed = rows, cols

    def place(self, ids: torch.Tensor) -> None:
        m, dev = len(ids), ids.device
        if self.fixed is None:
            rows, cols = torch.randint(0, self.rows, (m,), device=dev), torch.randint(0, self.cols, (m,), device=dev)
        else:
            rows, cols = self.fixed[0][ids], self.fixed[1][ids]
        origin = self.terrain.terrain_origins[rows, cols].clone()
        origin[:, :2] += _draw(self.region[cols])
        origin[:, 2] = torch.where(self.flat_col[cols], origin[:, 2], self.ground.footprint(origin[:, :2]))
        self.terrain.env_origins[ids] = origin
        self.terrain.terrain_levels[ids], self.terrain.terrain_types[ids] = rows, cols
        self.row[ids], self.col[ids] = rows, cols
        self.flat[ids] = self.flat_col[cols]
        self.on_stairs[ids] = self.stairs_col[cols]


def _region(sub, family: str) -> tuple[float, float, float, float, float]:
    """Where a column's robots spawn, from its origin: x0, x1, y0, y1 and the box around the origin to keep out of."""
    if family == "flat":
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if family in terrain_mod.PRINTED_STAIRS:
        last = sub.spawn_gap + float(sub.stairs()[2].sum())
        side = sub.width / 2 - STAIRS_SIDE
        return sub.spawn_gap - STAIRS_MARGIN, last + STAIRS_MARGIN, -side, side, 0.0
    reach = sub.size[0] / 2 - TILE_EDGE
    return -reach, reach, -reach, reach, PLATFORM_CLEAR


def _draw(r: torch.Tensor) -> torch.Tensor:
    """(m, 2) offsets uniform in each row's region, redrawn inside the keep-out box."""
    lo, hi, clear = r[:, [0, 2]], r[:, [1, 3]], r[:, 4]
    xy = lo + (hi - lo) * torch.rand(len(r), 2, device=r.device)
    while True:
        bad = (xy.abs().amax(-1) <= clear) & (clear > 0.0)
        if not bool(bad.any()):
            return xy
        xy[bad] = (lo + (hi - lo) * torch.rand(len(r), 2, device=r.device))[bad]
