"""Terrain for SKU2 (hip height 70 mm, foot sphere r 4.9 mm): sub-terrains for mjlab's terrain generator.

- ``low_poly_heights``: a low-poly relief (Delaunay triangles over a jittered grid, random heights), optionally
  periodic (opposite edges equal).
- ``LowPolyRoughCfg``: that relief as a sub-terrain, amplitude rising and feature size falling with difficulty, inside
  a flat border; ``RaisedReliefCfg``: relief on a plinth around a floor-level spawn pit with vertical sides.
- ``StairGridCfg``: straight stairs of solid boxes (a given layout, or a riser per row); ``JitterPyramidStairsCfg``:
  square stairs around the spawn with every riser and tread drawn; ``NarrowFlightCfg``: narrow straight flights.
- ``rough_basic_sub_terrains`` and the presets built on it (``rough``, ``rough_steps``, ``stairs``, ``stairs_steep``,
  ``stairs_narrow``): mjlab's families rescaled to SKU2 plus the low-poly relief, the raised relief and the stairs.

Heightfields here follow MuJoCo's layout: row index along y, column index along x (mjlab's own heightfield terrains
pass x as rows, which transposes them; harmless for random terrain). Heights are kept as floats (no vertical
quantisation).
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

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

TILE_SIZE = 1.6               # m
SPAWN_FOOTPRINT = (0.22, 0.14)  # m; the robot's footprint at the stand, plus margin (spawn above its highest point)


# --------------------------------------------------------------------------------------------------------------------
# Surfaces
# --------------------------------------------------------------------------------------------------------------------

def low_poly_heights(rng: np.random.Generator, size: tuple[float, float], pitch: float, amplitude: float,
                     feature: float, periodic: bool = False, slope_deg: float = 0.0) -> np.ndarray:
    """A low-poly relief sampled on a grid: (ny, nx) heights in metres, minimum 0.

    Points on a grid of spacing ``feature`` jittered by ±0.35·feature get heights uniform in [0, amplitude]; the
    surface is their Delaunay triangulation (piecewise planar). ``periodic`` makes opposite edges equal (the surface
    repeats). ``slope_deg`` adds a global incline along x.
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
    step up of ``edge`` (5 -> 25 mm with difficulty) plus the relief."""
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


STAIR_RISERS_MM = tuple(range(10, 75, 5))      # the stair grid's risers, one per row: 10, 15, ..., 70 mm
STAIR_TREADS_MM = (50, 80, 110, 140, 200)       # its treads, one per column
STAIR_LAYOUTS: dict = {}  # (tag, row) -> the generated stairs, relative to the spawn: filled by StairGridCfg.function


@dataclass(kw_only=True)
class StairGridCfg(SubTerrainCfg):
    """Straight stairs across the whole tile width, built from solid boxes standing on z = 0 (vertical faces, as a
    built step; a heightfield would ramp each riser over one cell): a flat approach, ``steps`` risers, a landing to the
    tile's far edge. The robot spawns ``spawn_gap`` before the first riser facing +x.

    The riser comes from the row: ``row = floor(difficulty * len(risers))`` (mjlab passes ``(row + U) / num_rows``, so
    the generator's ``num_rows`` must equal ``len(risers)``). ``direction`` "climb" starts on the floor and walks up;
    "descend" starts on a platform at the stairs' full height and walks down to the floor. ``jitter`` > 0 makes each
    riser ``h * U(1 - jitter, 1 + jitter)`` and each tread ``tread * U(0.8, 1.2)`` clamped to ``tread_clamp``, drawn
    from the generator's rng (reproducible by the terrain seed). ``layout`` = ((riser, tread), ...) in m builds a measured
    staircase instead (the last tread is ignored: the landing runs to the tile's edge unless
    ``landing`` is set). ``width`` (m) makes every tread box that wide, centred on the spawn line, on the full-width
    floor: a narrow flight with side drops of each tread's height; ``landing`` (m) ends the top box that far past the
    last riser (a drop beyond). Each tile's stairs are recorded in ``STAIR_LAYOUTS[(tag, row)]`` for scoring."""
    tread: float = 0.08
    risers: tuple = tuple(h / 1000.0 for h in STAIR_RISERS_MM)
    steps: int = 4
    direction: str = "climb"
    jitter: float = 0.0
    tread_clamp: tuple[float, float] = (0.05, 0.20)
    approach: float = 0.45          # the first riser's distance from the tile's near edge (m)
    spawn_gap: float = 0.15         # the robot's centre this far before the first riser
    floor_thickness: float = 0.05
    layout: tuple | None = None
    width: float | None = None      # m: the treads' width (None: the whole tile)
    landing: float | None = None    # m: the top tread's length past the last riser (None: to the tile's edge)
    tag: str = "stairs"

    def stairs(self, difficulty: float, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray]:
        """(row, rises (steps,), treads between risers (steps - 1,)) in m."""
        if self.layout is not None:      # one row (the generator's num_rows = 1)
            row = 0
            rises = np.array([r for r, _ in self.layout], dtype=float)
            treads = np.array([t for _, t in self.layout[:-1]], dtype=float)
            return row, rises, treads
        row = min(len(self.risers) - 1, int(difficulty * len(self.risers)))
        rises = np.full(self.steps, float(self.risers[row]))
        treads = np.full(self.steps - 1, float(self.tread))
        if self.jitter > 0:
            rises = rises * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter, self.steps)
            treads = np.clip(treads * rng.uniform(0.8, 1.2, self.steps - 1), *self.tread_clamp)
        return row, rises, treads

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        row, rises, treads = self.stairs(difficulty, rng)
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


def stair_grid_sub_terrains(size: float = TILE_SIZE, direction: str = "climb", jitter: float = 0.0,
                            treads_mm: tuple = STAIR_TREADS_MM, steps: int = 4) -> dict[str, SubTerrainCfg]:
    """A stair grid: one column per tread, one row per riser (``STAIR_RISERS_MM``)."""
    out = {}
    for t in treads_mm:
        tag = f"{direction}{'_irr' if jitter else ''}_t{t}"
        out[tag] = StairGridCfg(proportion=1.0 / len(treads_mm), size=(size, size), tread=t / 1000.0,
                                direction=direction, jitter=jitter, steps=steps, tag=tag)
    return out


@dataclass(kw_only=True)
class JitterPyramidStairsCfg(SubTerrainCfg):
    """Square stairs around the spawn, every riser and tread drawn (training's stairs for uneven steps). "climb": a pit,
    the spawn on its floor, rings rising outward to the border; "descend": a pyramid, the spawn on its top, rings
    falling outward to the floor. Either way every heading meets stairs. Per tile: the nominal riser
    ``riser_range[0] + difficulty * (riser_range[1] - riser_range[0])``, a nominal tread U(``tread_range``); per step
    the riser ``nominal * U(1 - jitter, 1 + jitter)`` and the tread ``nominal * U(0.8, 1.2)`` clamped to
    ``tread_range``; rings are added while they fit between the platform and the border. Solid boxes from z = 0 (frames
    of four boxes for the pit): vertical faces."""
    riser_range: tuple[float, float] = (0.005, 0.060)
    tread_range: tuple[float, float] = (0.05, 0.20)
    jitter: float = 0.3
    direction: str = "climb"
    platform_width: float = 0.40
    border_width: float = 0.10

    def steps(self, difficulty: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(rises, treads) from the platform outward, in m."""
        h = self.riser_range[0] + difficulty * (self.riser_range[1] - self.riser_range[0])
        t_nom = rng.uniform(*self.tread_range)
        room = (self.size[0] - 2 * self.border_width - self.platform_width) / 2
        rises, treads = [], []
        while True:
            t = float(np.clip(t_nom * rng.uniform(0.8, 1.2), *self.tread_range))
            if sum(treads) + t > room + 1e-9:
                break
            treads.append(t)
            rises.append(h * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter))
        rises.append(h * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter))     # the border's riser
        return np.array(rises), np.array(treads)

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        rises, treads = self.steps(difficulty, rng)
        self.last = (rises, treads)          # the latest tile's steps (for tests)
        # the outermost riser's half-size, per tile in generation order (the generator builds one column's rows in
        # order, so the last num_rows entries are rows 0..num_rows-1): where the stairs end (terrain_task.stair_exit)
        self.__dict__.setdefault("extents", []).append(float(self.platform_width / 2 + treads.sum()))
        sx = self.size[0]
        c = sx / 2
        half = self.platform_width / 2 + np.concatenate([[0.0], np.cumsum(treads)])   # each riser's half-size
        body, geoms = spec.body("terrain"), []
        g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[c, c, 0.025], pos=[c, c, -0.025],
                          rgba=[0.5, 0.5, 0.5, 1.0])
        geoms.append(TerrainGeometry(geom=g))
        if self.direction == "descend":      # nested solid squares, the inner the taller
            tops = np.cumsum(rises[::-1])[::-1]                                          # top inside each riser
            for a, top in zip(half, tops):
                g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[a, a, top / 2], pos=[c, c, top / 2],
                                  rgba=[0.55, 0.40, 0.30, 1.0])
                geoms.append(TerrainGeometry(geom=g))
            origin = np.array([c, c, float(tops[0])])
        elif self.direction == "climb":      # nested square frames from each riser to the tile's edge, the outer taller
            tops = np.cumsum(rises)                                                      # top outside each riser
            for a, top in zip(half, tops):
                w = c - a                                                                # the frame's width
                for pos, size in (([c, c + a + w / 2], [c, w / 2]), ([c, c - a - w / 2], [c, w / 2]),   # +-y strips
                                  ([c + a + w / 2, c], [w / 2, a]), ([c - a - w / 2, c], [w / 2, a])):  # +-x strips
                    g = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size + [top / 2], pos=pos + [top / 2],
                                      rgba=[0.55, 0.40, 0.30, 1.0])
                    geoms.append(TerrainGeometry(geom=g))
            origin = np.array([c, c, 0.0])
        else:
            raise ValueError(f"direction {self.direction!r}")
        return TerrainOutput(origin=origin, geometries=geoms)


@dataclass(kw_only=True)
class NarrowFlightCfg(StairGridCfg):
    """Training's narrow straight flights (modelled on a small built staircase): ``StairGridCfg`` climbs with, per tile,
    4-5 risers of ``riser_range[0] + difficulty * (riser_range[1] - riser_range[0])`` each times U(1 -+ ``jitter``), a
    nominal tread U(``tread_range``) times U(0.9, 1.1) per step (clamped), a width U(``width_range``) and a landing
    U(``landing_range``): drops at the sides and past the landing. Each tile's flight, relative to its spawn, is
    appended to ``flights`` in generation order (one column's rows in order: the last num_rows entries are rows
    0..num_rows-1) for ``terrain_task.stair_exit`` and ``side_exit``."""
    riser_range: tuple[float, float] = (0.010, 0.035)
    tread_range: tuple[float, float] = (0.045, 0.080)
    width_range: tuple[float, float] = (0.17, 0.30)
    landing_range: tuple[float, float] = (0.20, 0.40)
    steps_range: tuple[int, int] = (4, 5)
    jitter: float = 0.15
    tag: str = "narrow"

    def stairs(self, difficulty: float, rng: np.random.Generator) -> tuple[int, np.ndarray, np.ndarray]:
        n = int(rng.integers(self.steps_range[0], self.steps_range[1] + 1))
        h = self.riser_range[0] + difficulty * (self.riser_range[1] - self.riser_range[0])
        rises = h * rng.uniform(1.0 - self.jitter, 1.0 + self.jitter, n)
        t_nom = rng.uniform(*self.tread_range)
        treads = np.clip(t_nom * rng.uniform(0.9, 1.1, n - 1), *self.tread_range)
        self.width = float(rng.uniform(*self.width_range))
        self.landing = float(rng.uniform(*self.landing_range))
        return 0, rises, treads

    def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
        out = super().function(difficulty, spec, rng)
        L = STAIR_LAYOUTS[(self.tag, 0)]
        self.__dict__.setdefault("flights", []).append(
            {"x_first": float(L["risers_x"][0]), "x_last": float(L["risers_x"][-1]), "width": self.width,
             "landing": self.landing, "rises": L["rises"], "treads": L["treads"]})
        return out


def stairs_narrow_sub_terrains(size: float = TILE_SIZE, narrow: float = 0.20, pits: float = 0.10,
                               riser_range: tuple[float, float] | None = None) -> dict[str, SubTerrainCfg]:
    """``stairs`` with its jittered climbs (30 %) split into narrow straight flights (``narrow``) and the square pits
    (``pits``); the descents and the rough families unchanged. ``riser_range`` (m, nominal at difficulty 0 and 1)
    replaces the narrow flights' default (0.010, 0.035)."""
    subs = stairs_sub_terrains(size)
    subs["jstairs_climb"].proportion = pits
    subs["narrow"] = NarrowFlightCfg(proportion=narrow, size=(size, size),
                                     **({"riser_range": tuple(riser_range)} if riser_range else {}))
    return subs


def stairs_sub_terrains(size: float = TILE_SIZE, stairs: float = 0.50, climb: float = 0.30) -> dict[str, SubTerrainCfg]:
    """``rough``'s families without its stairs, rescaled to ``1 - stairs``; jittered stairs (risers 5 -> 60 mm with
    difficulty, ±30 % per step, treads 50-200 mm) climbing ``climb`` and descending ``stairs - climb``."""
    subs = rough_sub_terrains(size)
    for k in ("stairs_up", "stairs_down"):
        del subs[k]
    rest = sum(c.proportion for c in subs.values())
    for c in subs.values():
        c.proportion *= (1.0 - stairs) / rest
    sz = (size, size)
    subs["jstairs_climb"] = JitterPyramidStairsCfg(proportion=climb, size=sz, direction="climb")
    subs["jstairs_descend"] = JitterPyramidStairsCfg(proportion=stairs - climb, size=sz, direction="descend")
    return subs


def stairs_steep_sub_terrains(size: float = TILE_SIZE) -> dict[str, SubTerrainCfg]:
    """Aimed at a staircase of 25 mm risers on 50 mm treads: ``rough``'s families without its stairs at
    40 %; steep jittered stairs (nominal riser 15 -> 35 mm with difficulty, ±15 % per step, treads 45-65 mm)
    climbing 40 % and descending 20 %."""
    subs = stairs_sub_terrains(size, stairs=0.60, climb=0.40)
    for k in ("jstairs_climb", "jstairs_descend"):
        c = subs[k]
        c.riser_range, c.tread_range, c.jitter = (0.015, 0.035), (0.045, 0.065), 0.15
    return subs


def rough_sub_terrains(size: float = TILE_SIZE) -> dict[str, SubTerrainCfg]:
    """``rough_basic``'s mix with relief on a raised plinth (vertical edges, 5 -> 25 mm) taking 15 of the relief's
    40 %."""
    subs = rough_basic_sub_terrains(size)
    subs["low_poly"].proportion = 0.25
    subs["raised_relief"] = RaisedReliefCfg(proportion=0.15, size=(size, size))
    return subs


def rough_steps_sub_terrains(size: float = TILE_SIZE, steps: float = 0.60) -> dict[str, SubTerrainCfg]:
    """``rough``'s mix with the stairs (up and down, 5-30 mm) taking ``steps`` of the spawns; the other families
    keep their relative shares."""
    subs = rough_sub_terrains(size)
    stairs = ("stairs_up", "stairs_down")
    rest = sum(c.proportion for k, c in subs.items() if k not in stairs)
    for k, c in subs.items():
        c.proportion = steps / len(stairs) if k in stairs else c.proportion * (1.0 - steps) / rest
    return subs


def rough_basic_sub_terrains(size: float = TILE_SIZE) -> dict[str, SubTerrainCfg]:
    """The base training mix, every metre-valued field sized for SKU2. Proportions are spawn weights."""
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


PRESETS = {"rough_basic": rough_basic_sub_terrains, "rough": rough_sub_terrains,
           "rough_steps": rough_steps_sub_terrains, "stairs": stairs_sub_terrains,
           "stairs_steep": stairs_steep_sub_terrains, "stairs_narrow": stairs_narrow_sub_terrains}
