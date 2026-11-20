"""Terrain for the robot (hip height 70 mm, foot sphere r 5.5 mm): sub-terrains for mjlab's terrain generator.

- ``low_poly_heights``: a low-poly relief (Delaunay triangles over a jittered grid, random heights). ``LowPolyRoughCfg``:
  that relief as a sub-terrain, amplitude rising and feature size falling with difficulty, inside a flat border.
- ``RaisedReliefCfg``: the relief on a raised plinth with vertical edges.
- ``TileStripCfg``: a measured printed tile (``envs/data/terrain_tile1_hmap.npz``) laid on a flat floor as on the robot's test
  ground, with the floor-to-tile edges.
- ``StairGridCfg``: a measured staircase built from solid boxes.
- ``rough_sub_terrains``: mjlab's families rescaled to this robot plus the reliefs.
- ``family_columns``: chosen families, rough or printed stairs, one generator column each.

Heightfields here follow MuJoCo's layout: row index along y, column index along x (mjlab's own heightfield terrains
pass x as rows, which transposes them; harmless for random terrain, not for a measured tile). Heights are kept as
floats (no vertical quantisation).
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjlab.terrains.heightfield_terrains import (
    HfDiscreteObstaclesTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    color_by_height,
)
from mjlab.terrains.primitive_terrains import (
    BoxFlatTerrainCfg,
    BoxInvertedPyramidStairsTerrainCfg,
    BoxPyramidStairsTerrainCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

from mp4_sim import paths

TILE_SIZE = 1.6               # m; a multiple of the printed 0.2 m tile
SPAWN_FOOTPRINT = (0.22, 0.14)  # m; the robot's footprint at the stand, plus margin (spawn above its highest point)
LEVELS = 10                   # difficulty levels: level k has difficulty in [k, k + 1) / LEVELS, however many are built
PRINTED_STAIRS = {"stairs4": 4, "stairs8": 8}   # steps
PRINTED_RISER, PRINTED_TREAD, PRINTED_WIDTH = 0.025, 0.05, 0.4     # m


# --------------------------------------------------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------------------------------------------------

def low_poly_heights(rng: np.random.Generator, size: tuple[float, float], pitch: float, amplitude: float,
                     feature: float, periodic: bool = False, slope_deg: float = 0.0) -> np.ndarray:
    """A low-poly relief sampled on a grid: (ny, nx) heights in metres, minimum 0.

    Points on a grid of spacing ``feature`` jittered by ±0.35·feature get heights uniform in [0, amplitude]; the
    surface is their Delaunay triangulation (piecewise planar). ``periodic`` makes opposite edges equal (tiles chain
    seamlessly). ``slope_deg`` adds a global incline along x.
    """
    sx, sy = size
    nx_pts, ny_pts = max(2, int(round(sx / feature))), max(2, int(round(sy / feature)))
    gx, gy = np.meshgrid(np.arange(nx_pts) * sx / nx_pts, np.arange(ny_pts) * sy / ny_pts)
    jitter = rng.uniform(-0.35, 0.35, size=(2,) + gx.shape) * feature
    px, py = (gx + jitter[0]).ravel(), (gy + jitter[1]).ravel()
    pz = rng.uniform(0.0, amplitude, size=px.size)
    if periodic:  # copies shifted by the tile size on every side: the surface repeats exactly
        sh = [(dx, dy) for dx in (-sx, 0.0, sx) for dy in (-sy, 0.0, sy)]
        px = np.concatenate([px + dx for dx, _ in sh])
        py = np.concatenate([py + dy for _, dy in sh])
        pz = np.tile(pz, len(sh))
    else:  # the corners and edges pinned at mid height, so the hull covers the tile
        edge = np.linspace(0.0, 1.0, max(nx_pts, ny_pts) + 1)
        ex = np.concatenate([edge * sx, edge * sx, np.zeros_like(edge), np.full_like(edge, sx)])
        ey = np.concatenate([np.zeros_like(edge), np.full_like(edge, sy), edge * sy, edge * sy])
        px, py = np.concatenate([px, ex]), np.concatenate([py, ey])
        pz = np.concatenate([pz, rng.uniform(0.0, amplitude, size=ex.size)])
    interp = LinearNDInterpolator(Delaunay(np.column_stack([px, py])), pz)
    xs = np.arange(int(round(sx / pitch)) + 1) * pitch
    ys = np.arange(int(round(sy / pitch)) + 1) * pitch
    X, Y = np.meshgrid(xs, ys)
    h = interp(X, Y)
    h = np.where(np.isfinite(h), h, np.nanmean(h))
    h = h + np.tan(np.radians(slope_deg)) * X
    return (h - h.min()).astype(np.float64)


def flat_border(h: np.ndarray, pitch: float, border: float, ramp: float = 0.0) -> np.ndarray:
    """Force a flat band of width ``border`` at the edges to height 0, blending linearly over ``ramp`` inside it."""
    ny, nx = h.shape
    yy, xx = np.mgrid[0:ny, 0:nx] * pitch
    d = np.minimum.reduce([xx, yy, (nx - 1) * pitch - xx, (ny - 1) * pitch - yy])
    w = np.clip((d - border) / ramp, 0.0, 1.0) if ramp > 0 else (d >= border).astype(float)
    return h * w


def load_tile(path: Path | str, pitch: float, which: str = "height_mm") -> np.ndarray:
    """A measured tile heightmap (``view_mesh.py --heightmap``), resampled to ``pitch`` (m), heights in metres.

    The npz grid is rows = y, columns = x, in mm, at its own pitch; heights are the printed part's top surface above
    the print bed (the floor the tile lies on).
    """
    d = np.load(path)
    h = np.asarray(d[which], dtype=np.float64) / 1000.0
    src = float(d["pitch_mm"]) / 1000.0
    h = np.where(np.isfinite(h), h, np.nanmin(h))
    ny, nx = h.shape
    xs, ys = np.arange(nx) * src, np.arange(ny) * src
    nxo, nyo = int(xs[-1] / pitch) + 1, int(ys[-1] / pitch) + 1
    xi, yi = np.arange(nxo) * pitch / src, np.arange(nyo) * pitch / src
    from scipy.ndimage import map_coordinates
    Y, X = np.meshgrid(yi, xi, indexing="ij")
    return map_coordinates(h, [Y, X], order=1, mode="nearest")


def strip_heights(tile: np.ndarray, n_long: int, n_wide: int, layout: str) -> np.ndarray:
    """Tiles laid n_long along x and n_wide along y. ``layout``: 'repeat', 'mirror_x' (every other tile along x mirrored
    in x), 'mirror_y', 'mirror_xy'."""
    rows = []
    for j in range(n_wide):
        row = []
        for i in range(n_long):
            t = tile
            if layout in ("mirror_x", "mirror_xy") and i % 2:
                t = t[:, ::-1]
            if layout in ("mirror_y", "mirror_xy") and j % 2:
                t = t[::-1, :]
            row.append(t if i == n_long - 1 else t[:, :-1])          # neighbours share the joining sample
        r = np.concatenate(row, axis=1)
        rows.append(r if j == n_wide - 1 else r[:-1])
    return np.concatenate(rows, axis=0)


# --------------------------------------------------------------------------------------------------------------------
# Sub-terrains
# --------------------------------------------------------------------------------------------------------------------

def add_hfield(spec: mujoco.MjSpec, h: np.ndarray, pitch: float, base: float, origin_xy=(0.0, 0.0)) -> TerrainGeometry:
    """Add heights h (ny, nx, metres, min >= 0) as one heightfield whose lower-left corner sits at origin_xy."""
    ny, nx = h.shape
    hmax = float(max(h.max(), 1e-4))
    uid = uuid.uuid4().hex
    hf = spec.add_hfield(name=f"hfield_{uid}", size=[(nx - 1) * pitch / 2, (ny - 1) * pitch / 2, hmax, base],
                         nrow=ny, ncol=nx, userdata=(h / hmax).ravel().astype(np.float32).tolist())
    material = color_by_height(spec, h, uid, h / hmax)  # texture rows follow the heightfield's rows
    g = spec.body("terrain").add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hf.name, material=material,
                                      pos=[origin_xy[0] + (nx - 1) * pitch / 2, origin_xy[1] + (ny - 1) * pitch / 2,
                                           0.0])
    return TerrainGeometry(geom=g, hfield=hf)


def _wall_line(body, out: list, heights: np.ndarray, start: int, centre: float, along_x: bool, pitch: float,
               merge_m: float) -> None:
    """Boxes standing on z = 0, one per run of ``heights`` within ``merge_m`` (each box as tall as its run's highest
    sample), one cell (``pitch``) thick, centred at ``centre`` (m) across the line; the line runs along x from sample
    ``start`` when ``along_x``, else along y."""
    i = 0
    while i < len(heights):
        j = i
        while j + 1 < len(heights) and abs(heights[j + 1] - heights[i]) <= merge_m:
            j += 1
        hz = float(max(heights[i:j + 1]))
        if hz > 1e-4:
            mid, half = (start + (i + j) / 2) * pitch, (j - i + 1) * pitch / 2
            pos, size = ([mid, centre, hz / 2], [half, pitch / 2, hz / 2]) if along_x else \
                ([centre, mid, hz / 2], [pitch / 2, half, hz / 2])
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=[0.55, 0.55, 0.5, 1.0])
            out.append(TerrainGeometry(geom=g))
        i = j + 1


def edge_walls(spec: mujoco.MjSpec, h: np.ndarray, pitch: float, r0: int, c0: int, ny: int, nx: int,
               merge_m: float = 0.001) -> list[TerrainGeometry]:
    """Vertical faces for a raised block of heights ``h[r0:r0+ny, c0:c0+nx]`` standing on a zero floor: boxes filling
    the heightfield cell just outside each edge to the edge's height. A heightfield alone ramps from the floor to the
    edge over one cell (61-73 degrees at 5 mm cells for a 9-18 mm edge); the printed tiles' sides are vertical."""
    body, out = spec.body("terrain"), []
    for col, dx in ((c0, -1), (c0 + nx - 1, 1)):                     # the near and far edges (constant x)
        _wall_line(body, out, h[r0:r0 + ny, col], r0, (col + 0.5 * dx) * pitch, False, pitch, merge_m)
    for row, dy in ((r0, -1), (r0 + ny - 1, 1)):                     # the side edges (constant y)
        _wall_line(body, out, h[row, c0:c0 + nx], c0, (row + 0.5 * dy) * pitch, True, pitch, merge_m)
    return out


def pit_walls(spec: mujoco.MjSpec, h: np.ndarray, pitch: float, r0: int, c0: int, ny: int, nx: int,
              merge_m: float = 0.001) -> list[TerrainGeometry]:
    """Vertical faces around a zero-height pit ``h[r0:r0+ny, c0:c0+nx]`` in raised ground: boxes filling the cell
    just inside each side of the pit to the height of the raised sample beyond it."""
    body, out = spec.body("terrain"), []
    for col, dx in ((c0 - 1, 1), (c0 + nx, -1)):                     # the raised columns either side, in x
        _wall_line(body, out, h[r0:r0 + ny, col], r0, (col + 0.5 * dx) * pitch, False, pitch, merge_m)
    for row, dy in ((r0 - 1, 1), (r0 + ny, -1)):                     # the raised rows either side, in y
        _wall_line(body, out, h[row, c0:c0 + nx], c0, (row + 0.5 * dy) * pitch, True, pitch, merge_m)
    return out


def spawn_height(h: np.ndarray, pitch: float, cx: float, cy: float) -> float:
    """The highest point of the surface under the robot's footprint centred at (cx, cy)."""
    hx, hy = SPAWN_FOOTPRINT[0] / 2, SPAWN_FOOTPRINT[1] / 2
    i0, i1 = int(max(0, (cy - hy) / pitch)), int(min(h.shape[0] - 1, (cy + hy) / pitch)) + 1
    j0, j1 = int(max(0, (cx - hx) / pitch)), int(min(h.shape[1] - 1, (cx + hx) / pitch)) + 1
    return float(h[i0:i1, j0:j1].max())


@dataclass(kw_only=True)
class LowPolyRoughCfg(SubTerrainCfg):
    amplitude_range: tuple[float, float] = (0.005, 0.030)   # peak to valley, m
    feature_range: tuple[float, float] = (0.08, 0.04)       # point spacing at difficulty 0 and 1, m
    pitch: float = 0.005
    border_width: float = 0.10
    border_ramp: float = 0.10
    base_thickness: float = 0.02
    platform_width: float = 0.30                            # a flat square in the middle to spawn on

    def heights(self, difficulty: float, rng: np.random.Generator) -> np.ndarray:
        a = self.amplitude_range[0] + difficulty * (self.amplitude_range[1] - self.amplitude_range[0])
        f = self.feature_range[0] + difficulty * (self.feature_range[1] - self.feature_range[0])
        h = low_poly_heights(rng, self.size, self.pitch, a, f)
        h = flat_border(h, self.pitch, self.border_width, self.border_ramp)
        ny, nx = h.shape
        c = (np.array([nx, ny]) - 1) * self.pitch / 2
        yy, xx = np.mgrid[0:ny, 0:nx] * self.pitch
        inner = (np.abs(xx - c[0]) < self.platform_width / 2) & (np.abs(yy - c[1]) < self.platform_width / 2)
        h[inner] = np.median(h[inner])  # a flat spawn platform at the local level
        return h - h.min()

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        h = self.heights(difficulty, rng)
        geom = add_hfield(spec, h, self.pitch, self.base_thickness)
        cx, cy = self.size[0] / 2, self.size[1] / 2
        return TerrainOutput(origin=np.array([cx, cy, spawn_height(h, self.pitch, cx, cy)]), geometries=[geom])


@dataclass(kw_only=True)
class RaisedReliefCfg(SubTerrainCfg):
    """Low-poly relief on a plinth around a floor-level spawn pit with vertical sides: every way out of the pit is a
    step up of ``edge`` (5 -> 25 mm with difficulty) plus the relief, as onto the printed tiles."""
    edge_range: tuple[float, float] = (0.005, 0.025)       # the plinth's height above the pit floor, m
    amplitude_range: tuple[float, float] = (0.005, 0.020)  # the relief on it, peak to valley, m
    feature_range: tuple[float, float] = (0.08, 0.05)
    pitch: float = 0.005
    pit_width: float = 0.30
    border_width: float = 0.10
    border_ramp: float = 0.10
    base_thickness: float = 0.02

    def heights(self, difficulty: float, rng: np.random.Generator):
        lerp = lambda r: r[0] + difficulty * (r[1] - r[0])   # noqa: E731
        h = lerp(self.edge_range) + low_poly_heights(rng, self.size, self.pitch, lerp(self.amplitude_range),
                                                     lerp(self.feature_range))
        h = flat_border(h, self.pitch, self.border_width, self.border_ramp)
        ny, nx = h.shape
        n = int(round(self.pit_width / self.pitch))
        r0, c0 = (ny - n) // 2, (nx - n) // 2
        h[r0:r0 + n, c0:c0 + n] = 0.0
        return h, (r0, c0, n, n)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        h, pit = self.heights(difficulty, rng)
        geoms = [add_hfield(spec, h, self.pitch, self.base_thickness)] + pit_walls(spec, h, self.pitch, *pit)
        return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]), geometries=geoms)


@dataclass(kw_only=True)
class TileStripCfg(SubTerrainCfg):
    """A strip of printed tiles on a flat floor, entered from the floor at x = ``start_x`` (the spawn faces +x)."""
    tile_npz: str = str(paths.ENV_DATA / "terrain_tile1_hmap.npz")
    layout: str = "mirror_xy"      # every edge a mirror line (the printed pair: a tile and its mirror image, laid
                                   # so that each neighbour is the mirror across the shared edge): seamless joins
    n_long: int = 4
    n_wide: int = 2
    pitch: float = 0.005           # 0.25 mm mean / 1.1 mm p99 off the measured surface; at 2.5 mm, shins and hip
                                   # cans scraping the relief exceed mujoco_warp's 50-prism cap and its warnings stall runs
    start_x: float = 0.40          # the strip's near edge, from the sub-terrain's corner
    spawn_gap: float = 0.15        # the robot's centre this far before the strip
    base_thickness: float = 0.02
    walls: bool = False            # vertical faces at the strip's edges, as on the printed tiles (``edge_walls``)

    def heights(self) -> tuple[np.ndarray, tuple[float, float]]:
        tile = load_tile(self.tile_npz, self.pitch)
        strip = strip_heights(tile, self.n_long, self.n_wide, self.layout)
        nx, ny = int(round(self.size[0] / self.pitch)) + 1, int(round(self.size[1] / self.pitch)) + 1
        h = np.zeros((ny, nx))
        i0 = int(round((self.size[1] / 2) / self.pitch)) - strip.shape[0] // 2
        j0 = int(round(self.start_x / self.pitch))
        h[i0:i0 + strip.shape[0], j0:j0 + strip.shape[1]] = strip[:ny - i0, :nx - j0]
        self._block = (i0, j0, min(strip.shape[0], ny - i0), min(strip.shape[1], nx - j0))
        return h, (j0 * self.pitch, i0 * self.pitch)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        del difficulty, rng
        h, _ = self.heights()
        geoms = [add_hfield(spec, h, self.pitch, self.base_thickness)]
        if self.walls:
            geoms += edge_walls(spec, h, self.pitch, *self._block)
        cx, cy = self.start_x - self.spawn_gap, self.size[1] / 2
        return TerrainOutput(origin=np.array([cx, cy, spawn_height(h, self.pitch, cx, cy)]), geometries=geoms)


STAIR_LAYOUTS: dict = {}  # (tag, row) -> the generated stairs, relative to the spawn: filled by StairGridCfg.function


@dataclass(kw_only=True)
class StairGridCfg(SubTerrainCfg):
    """A measured staircase built from solid boxes standing on z = 0 (vertical faces, as a built step; a heightfield
    would ramp each riser over one cell): a flat approach, the risers of ``layout`` = ((riser, tread), ...) in m, a landing
    to the tile's far edge (the last tread is ignored). The robot spawns ``spawn_gap`` before the first riser facing +x.

    ``width`` (m) makes every tread box that wide, centred on the spawn line, on the full-width floor: a narrow print with
    side drops of each tread's height; ``landing`` (m) ends the top box that far past the last riser (a drop beyond). The
    stairs are recorded in ``STAIR_LAYOUTS[(tag, row)]`` for scoring."""
    direction: str = "climb"
    approach: float = 0.45          # the first riser's distance from the tile's near edge (m)
    spawn_gap: float = 0.15         # the robot's centre this far before the first riser
    floor_thickness: float = 0.05
    layout: tuple | None = None
    width: float | None = None      # m: the treads' width (None: the whole tile)
    landing: float | None = None    # m: the top tread's length past the last riser (None: to the tile's edge)
    tag: str = "stairs"

    def stairs(self) -> tuple[int, np.ndarray, np.ndarray]:
        """(row, rises (steps,), treads between risers (steps - 1,)) in m; one row (the generator's num_rows = 1)."""
        rises = np.array([r for r, _ in self.layout], dtype=float)
        treads = np.array([t for _, t in self.layout[:-1]], dtype=float)
        return 0, rises, treads

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        row, rises, treads = self.stairs()
        sx, sy = self.size
        xr = self.approach + np.concatenate([[0.0], np.cumsum(treads)])          # the risers' x in the tile
        x_end = sx if self.landing is None else min(sx, float(xr[-1]) + self.landing)
        edges = np.concatenate([[0.0], xr, [x_end]])                             # the tops' x extents
        wy = sy if self.width is None else self.width                            # the treads' width
        if self.direction == "climb":
            tops = np.concatenate([[0.0], np.cumsum(rises)])                     # floor, then each tread's top
        elif self.direction == "descend":
            tops = rises.sum() - np.concatenate([[0.0], np.cumsum(rises)])       # platform, ..., floor
        else:
            raise ValueError(f"direction {self.direction!r}")
        assert edges[-2] + 0.10 <= sx, "the stairs do not fit the tile"
        body, geoms = spec.body("terrain"), []
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx / 2, sy / 2, self.floor_thickness / 2],
                          pos=[sx / 2, sy / 2, -self.floor_thickness / 2], rgba=[0.5, 0.5, 0.5, 1.0])
        geoms.append(TerrainGeometry(geom=g))
        for i, top in enumerate(tops):
            if top <= 1e-6:
                continue
            x0, x1 = edges[i], edges[i + 1]
            shade = 0.45 + 0.35 * (i % 2)
            g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, wy / 2, top / 2],
                              pos=[(x0 + x1) / 2, sy / 2, top / 2], rgba=[shade, 0.35, 0.30, 1.0])
            geoms.append(TerrainGeometry(geom=g))
        spawn = np.array([self.approach - self.spawn_gap, sy / 2, float(tops[0])])
        STAIR_LAYOUTS[(self.tag, row)] = {"direction": self.direction, "risers_x": (xr - spawn[0]).tolist(),
                                          "rises": rises.tolist(), "treads": treads.tolist(),
                                          "spawn_z": float(tops[0]), "width": self.width, "landing": self.landing}
        return TerrainOutput(origin=spawn, geometries=geoms)


def _base_mix(size: float) -> dict[str, SubTerrainCfg]:
    """Every metre-valued field sized for this robot. Proportions are spawn weights."""
    sz = (size, size)
    return {
        "flat": BoxFlatTerrainCfg(proportion=0.10, size=sz),
        "low_poly": LowPolyRoughCfg(proportion=0.40, size=sz),
        "slope_up": HfPyramidSlopedTerrainCfg(proportion=0.10, size=sz, slope_range=(0.0, np.tan(np.radians(20))),
                                              platform_width=0.30, border_width=0.10, horizontal_scale=0.02,
                                              vertical_scale=0.001, inverted=False),
        "slope_down": HfPyramidSlopedTerrainCfg(proportion=0.10, size=sz, slope_range=(0.0, np.tan(np.radians(20))),
                                                platform_width=0.30, border_width=0.10, horizontal_scale=0.02,
                                                vertical_scale=0.001, inverted=True),
        "boxes": HfDiscreteObstaclesTerrainCfg(proportion=0.15, size=sz, obstacle_height_mode="choice",
                                               obstacle_width_range=(0.04, 0.12),
                                               obstacle_height_range=(0.005, 0.030), num_obstacles=120,
                                               platform_width=0.30, border_width=0.10, horizontal_scale=0.01,
                                               vertical_scale=0.001),
        "stairs_up": BoxPyramidStairsTerrainCfg(proportion=0.075, size=sz, step_height_range=(0.005, 0.030),
                                                step_width=0.08, platform_width=0.30, border_width=0.10),
        "stairs_down": BoxInvertedPyramidStairsTerrainCfg(proportion=0.075, size=sz,
                                                          step_height_range=(0.005, 0.030), step_width=0.08,
                                                          platform_width=0.30, border_width=0.10),
    }


# --------------------------------------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------------------------------------


def rough_sub_terrains(size: float = TILE_SIZE) -> dict[str, SubTerrainCfg]:
    """The base mix with relief on a raised plinth (vertical edges, 5 -> 25 mm) taking 15 of the relief's 40 %."""
    subs = _base_mix(size)
    subs["low_poly"].proportion = 0.25
    subs["raised_relief"] = RaisedReliefCfg(proportion=0.15, size=(size, size))
    return subs


def printed_stairs(steps: int, size: float = TILE_SIZE) -> StairGridCfg:
    """The printed flight, 400 mm wide, its top tread reaching the tile's far edge."""
    layout = ((PRINTED_RISER, PRINTED_TREAD),) * steps
    return StairGridCfg(size=(size, size), layout=layout, width=PRINTED_WIDTH, tag=f"stairs{steps}")


def family_of(column: str) -> str:
    return column.split(".")[0]


def family_columns(names, size: float = TILE_SIZE, relief_pitch: float | None = None) -> dict[str, SubTerrainCfg]:
    """A column per name, in order: a family of ``rough_sub_terrains`` or of ``PRINTED_STAIRS``. In curriculum mode a
    family is one column whatever its proportion, so a share is set by repeating the name; the repeats are named
    ``family.1``, ``family.2``, ... ``relief_pitch``: the reliefs' heightfield pitch (m)."""
    rough = rough_sub_terrains(size)
    out: dict[str, SubTerrainCfg] = {}
    seen: dict[str, int] = {}
    for name in names:
        cfg = printed_stairs(PRINTED_STAIRS[name], size) if name in PRINTED_STAIRS else copy.deepcopy(rough[name])
        if relief_pitch is not None and isinstance(cfg, (LowPolyRoughCfg, RaisedReliefCfg)):
            cfg.pitch = relief_pitch
        k = seen.get(name, 0)
        seen[name] = k + 1
        out[f"{name}.{k}" if k else name] = cfg
    return out
