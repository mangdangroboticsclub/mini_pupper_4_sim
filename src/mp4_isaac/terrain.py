"""The terrain objects the rough and stair policies were trained on, as Isaac prims.

``rough`` is the ``rough`` preset: a low-poly heightfield, scattered boxes, a
pyramid the robot climbs, a ramp, and a raised plinth with a floor-level pit.
``stairs`` is the ``stairs_narrow`` preset the stair policy finished on: a
narrow straight flight (risers 25 to 45 mm), a square climb pit, and a
pyramid the robot walks down. Sizes are the training ranges at mid difficulty,
with the flight's risers taken from that policy's ``--narrow-risers 0.025 0.045``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Training tile quantities, at difficulty 0.5 unless a policy pinned the range.
LOW_POLY_AMPLITUDE = 0.0175     # m, peak to valley (5 mm at difficulty 0, 30 mm at 1)
LOW_POLY_FEATURE = 0.06         # m, the relief's point spacing
RELIEF_EDGE = 0.015             # m, the plinth above its pit
RELIEF_AMPLITUDE = 0.0125
NARROW_RISER = 0.035            # m, midpoint of the stair policy's 25-45 mm
NARROW_TREAD = 0.060
NARROW_WIDTH = 0.22
NARROW_LANDING = 0.25
NARROW_STEPS = 5
STAIR_RISER = 0.015             # m, midpoint of the rough preset's 5-30 mm stairs
STAIR_TREAD = 0.08


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    color: tuple[float, float, float]


@dataclass
class Surface:
    """A triangle mesh in metres. Winding is counter-clockwise from above."""

    vertices: np.ndarray
    faces: np.ndarray
    color: tuple[float, float, float]


@dataclass
class Course:
    name: str
    spawn: tuple[float, float, float]
    camera_eye: tuple[float, float, float]
    camera_look: tuple[float, float, float]
    boxes: list[Box] = field(default_factory=list)
    surfaces: list[Surface] = field(default_factory=list)


def low_poly_heights(rng: np.random.Generator, size: tuple[float, float], pitch: float, amplitude: float,
                     feature: float) -> np.ndarray:
    """The training relief: Delaunay triangles over a jittered grid, sampled on ``pitch``."""
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay

    sx, sy = size
    nx_pts, ny_pts = max(2, int(round(sx / feature))), max(2, int(round(sy / feature)))
    gx, gy = np.meshgrid(np.arange(nx_pts) * sx / nx_pts, np.arange(ny_pts) * sy / ny_pts)
    jitter = rng.uniform(-0.35, 0.35, size=(2,) + gx.shape) * feature
    px, py = (gx + jitter[0]).ravel(), (gy + jitter[1]).ravel()
    pz = rng.uniform(0.0, amplitude, size=px.size)
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
    return (h - h.min()).astype(np.float64)


def _grid_surface(h: np.ndarray, pitch: float, origin: tuple[float, float], color) -> Surface:
    ny, nx = h.shape
    xs = origin[0] + np.arange(nx) * pitch
    ys = origin[1] + np.arange(ny) * pitch
    xx, yy = np.meshgrid(xs, ys)
    vertices = np.column_stack([xx.ravel(), yy.ravel(), h.ravel()]).astype(np.float64)
    faces = []
    for i in range(ny - 1):
        for j in range(nx - 1):
            a = i * nx + j
            b = a + 1
            c = a + nx
            d = c + 1
            faces.append((a, b, d))
            faces.append((a, d, c))
    return Surface(vertices, np.asarray(faces, dtype=np.int32), color)


def _pyramid(cx: float, cy: float, *, steps: int, riser: float, tread: float, platform: float,
             color: tuple[float, float, float]) -> list[Box]:
    """Solid nested squares. The middle is the tallest, so a robot walking in climbs."""
    boxes = []
    for i in range(steps):
        half = platform / 2 + (steps - 1 - i) * tread
        top = (i + 1) * riser
        shade = 0.42 + 0.08 * (i % 2)
        boxes.append(Box((cx, cy, top / 2), (2 * half, 2 * half, top), (shade, color[1], color[2])))
    return boxes


def _pit(cx: float, cy: float, *, steps: int, riser: float, tread: float, platform: float) -> list[Box]:
    """Square frames around a floor-level pit. The outside is the tallest."""
    boxes = []
    for i in range(steps):
        inner = platform / 2 + i * tread
        outer = inner + tread
        top = (steps - i) * riser
        shade = 0.40 + 0.08 * (i % 2)
        color = (shade, 0.32, 0.26)
        w = outer - inner
        for pos, size in (
            ((cx, cy + (inner + outer) / 2, top / 2), (2 * outer, w, top)),
            ((cx, cy - (inner + outer) / 2, top / 2), (2 * outer, w, top)),
            ((cx + (inner + outer) / 2, cy, top / 2), (w, 2 * inner, top)),
            ((cx - (inner + outer) / 2, cy, top / 2), (w, 2 * inner, top)),
        ):
            boxes.append(Box(pos, size, color))
    return boxes


def _flight(x0: float, *, steps: int, riser: float, tread: float, width: float, landing: float) -> list[Box]:
    """A narrow straight climb: ``steps`` risers, then a landing, drops at the sides."""
    rises = np.full(steps, riser)
    treads = np.full(steps - 1, tread)
    edges = np.concatenate([[x0], x0 + np.cumsum(treads), [x0 + treads.sum() + landing]])
    tops = np.cumsum(rises)
    boxes = []
    for i, top in enumerate(tops):
        x_a, x_b = float(edges[i]), float(edges[i + 1])
        shade = 0.45 + 0.12 * (i % 2)
        boxes.append(Box(((x_a + x_b) / 2, 0.0, top / 2), (x_b - x_a, width, top), (shade, 0.34, 0.28)))
    return boxes


def _ramp(x0: float, y: float, length: float, width: float, rise: float) -> Surface:
    """A wedge. The low edge is at ``x0`` and the high edge is ``rise`` above the floor."""
    y0, y1 = y - width / 2, y + width / 2
    x1 = x0 + length
    vertices = np.array(
        [
            [x0, y0, 0.0], [x0, y1, 0.0], [x1, y1, rise], [x1, y0, rise],
            [x0, y0, -0.02], [x0, y1, -0.02], [x1, y1, -0.02], [x1, y0, -0.02],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            (0, 3, 2), (0, 2, 1),
            (4, 6, 5), (4, 7, 6),
            (0, 4, 5), (0, 5, 1),
            (3, 2, 6), (3, 6, 7),
            (1, 5, 6), (1, 6, 2),
            (0, 3, 7), (0, 7, 4),
        ],
        dtype=np.int32,
    )
    return Surface(vertices, faces, (0.42, 0.44, 0.40))


def _obstacles(rng: np.random.Generator, x0: float, length: float, width: float, count: int) -> list[Box]:
    boxes = []
    for _ in range(count):
        w = float(rng.uniform(0.04, 0.12))
        h = float(rng.uniform(0.005, 0.030))
        x = float(rng.uniform(x0 + w / 2, x0 + length - w / 2))
        y = float(rng.uniform(-width / 2, width / 2))
        boxes.append(Box((x, y, h / 2), (w, w, h), (0.32, 0.32, 0.30)))
    return boxes


CROWD_GAP = (0.42, 0.36)  # metres between grid rows (x) and columns (y)
# The crowd stair field: one blue tread per rank, deep enough for a robot, the riser the policy climbs.
FIELD_TREAD = 0.42
FIELD_RISER = 0.035
FIELD_GAP_Y = 0.40
FIELD_BLUES = (
    (0.30, 0.52, 0.78),
    (0.18, 0.36, 0.64),
    (0.40, 0.60, 0.84),
    (0.22, 0.42, 0.70),
    (0.48, 0.66, 0.88),
)


def crowd_layout(n: int) -> tuple[np.ndarray, int, int]:
    """Robot centres in the ground plane. Row 0 is the front rank, toward +x.

    The last row is shorter when ``n`` does not fill a rectangle, and that row is centred.
    """
    n = max(1, int(n))
    if n == 1:
        return np.zeros((1, 2)), 1, 1
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    gap_x, gap_y = CROWD_GAP
    points = []
    placed = 0
    for row in range(rows):
        count = cols if row < rows - 1 else n - placed
        for col in range(count):
            points.append((-row * gap_x, (col - (count - 1) / 2.0) * gap_y))
        placed += count
    return np.asarray(points, dtype=float), rows, cols


def stair_formation(n: int) -> tuple[np.ndarray, int, int]:
    """Where each robot stands on the blue stair field: ``(x, y, surface z)``.

    A crowd is a filled rectangle, four across when there are enough robots, so the
    ranks recede up the stairs. The front rank is the lowest step.
    """
    n = max(1, int(n))
    cols = 4 if n >= 8 else max(1, int(np.ceil(np.sqrt(n))))
    rows = int(np.ceil(n / cols))
    points = []
    for row in range(rows):
        count = min(cols, n - row * cols)
        top = (row + 1) * FIELD_RISER
        for col in range(count):
            y = (col - (cols - 1) / 2.0) * FIELD_GAP_Y
            x = 0.10 + (row + 0.5) * FIELD_TREAD
            points.append((x, y, top))
    return np.asarray(points, dtype=float), rows, cols


def _stair_field(rows: int, cols: int) -> list[Box]:
    """Blue treads covering the formation, plus a few more steps ahead of the lead rank."""
    width = (cols - 1) * FIELD_GAP_Y + 0.36
    boxes = []
    for step in range(rows + 3):
        top = (step + 1) * FIELD_RISER
        x = 0.10 + (step + 0.5) * FIELD_TREAD
        boxes.append(Box(
            (x, 0.0, top / 2),
            (FIELD_TREAD, width, top),
            FIELD_BLUES[step % len(FIELD_BLUES)],
        ))
    return boxes


def build(name: str, seed: int = 4, lanes: int = 1) -> Course:
    """One playback course. ``flat`` is only the ground plane the scene already has."""
    if name == "flat":
        return Course("flat", (0.0, 0.0, 0.0), (0.45, -0.55, 0.28), (0.15, 0.0, 0.06))
    rng = np.random.default_rng(seed)
    if name == "rough":
        patch = (0.70, 0.50)
        h = low_poly_heights(rng, patch, 0.01, LOW_POLY_AMPLITUDE, LOW_POLY_FEATURE)
        relief = RELIEF_EDGE + low_poly_heights(rng, (0.55, 0.45), 0.01, RELIEF_AMPLITUDE, 0.07)
        pit = int(round(0.22 / 0.01))
        r0, c0 = (relief.shape[0] - pit) // 2, (relief.shape[1] - pit) // 2
        relief[r0:r0 + pit, c0:c0 + pit] = 0.0
        return Course(
            "rough",
            (0.0, 0.0, 0.0),
            (-0.15, -1.15, 0.55),
            (0.65, 0.0, 0.04),
            boxes=_obstacles(rng, 0.95, 0.55, 0.40, 10) + _pyramid(1.85, 0.0, steps=3, riser=STAIR_RISER, tread=STAIR_TREAD, platform=0.22, color=(0.50, 0.36, 0.28)),
            surfaces=[
                _grid_surface(h, 0.01, (0.15, -patch[1] / 2), (0.48, 0.50, 0.42)),
                _grid_surface(relief, 0.01, (0.20, 0.55), (0.55, 0.48, 0.38)),
                _ramp(0.25, -0.75, 0.45, 0.28, 0.45 * np.tan(np.radians(10.0))),
            ],
        )
    if name == "stairs":
        # One robot climbs the narrow flight. A crowd stands on a field of blue treads,
        # one rank per step, the way the formation recedes up the stairs.
        lanes = max(1, int(lanes))
        if lanes == 1:
            return Course(
                "stairs",
                (0.0, 0.0, 0.0),
                (-0.05, -1.05, 0.48),
                (0.45, 0.0, 0.06),
                boxes=_flight(
                    0.18, steps=NARROW_STEPS, riser=NARROW_RISER, tread=NARROW_TREAD,
                    width=NARROW_WIDTH, landing=NARROW_LANDING,
                )
                + _pit(0.45, 0.70, steps=3, riser=0.018, tread=0.07, platform=0.24)
                + _pyramid(0.45, -0.70, steps=3, riser=0.018, tread=0.07, platform=0.24, color=(0.50, 0.36, 0.28)),
            )
        slots, rows, cols = stair_formation(lanes)
        span_y = (cols - 1) * FIELD_GAP_Y
        # Beside the front rank, about as high as the body, looking up the field.
        eye = (float(slots[0, 0]) + 0.05, -(span_y / 2 + 0.55), 0.10)
        look = (float(slots[min(len(slots) - 1, cols * 2), 0]), 0.0, 0.06)
        return Course("stairs", (0.0, 0.0, 0.0), eye, look, boxes=_stair_field(rows, cols))
    raise ValueError(f"unknown terrain {name!r}; expected flat, rough, or stairs")


def assets(name: str, lanes: int = 1) -> dict:
    """Isaac prims for :func:`build`. Empty for ``flat``."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    course = build(name, lanes=lanes)
    out: dict = {}
    for i, box in enumerate(course.boxes):
        out[f"terrain_box_{i}"] = AssetBaseCfg(
            prim_path=f"/World/Terrain/box_{i}",
            spawn=sim_utils.CuboidCfg(
                size=box.size,
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=box.color),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=box.center),
        )
    for i, surface in enumerate(course.surfaces):
        out[f"terrain_mesh_{i}"] = AssetBaseCfg(
            prim_path=f"/World/Terrain/mesh_{i}",
            spawn=sim_utils.MeshFileCfg(
                mesh=sim_utils.MeshFileCfg.TriangleMeshCfg(
                    vertices=surface.vertices.tolist(),
                    faces=surface.faces.tolist(),
                ),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=surface.color),
            ),
        )
    return out
