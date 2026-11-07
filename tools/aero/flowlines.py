'''flowlines.py: streamlines from the panel method, as ribbons TouchDesigner can draw.

Imported by: aero/pipeline.py, touchdesigner/aero_td.py (through the pipeline), tests
Run directly: no

Two sets of lines, both on surfaces the projector actually lights:
  surface lines  follow the air across the model's own faces: seeds spread
                 evenly over the model, each traced up- and downstream by
                 walking from triangle to triangle along the surface velocity
  ground lines   the air just above the plate, parting round the model:
                 seeds on a rake upstream, traced through a velocity grid

Everything is in the plate frame, in mm. Like the surface pressure, the flow
is linear in the wind, so the ground grid stores a 2x2 matrix per cell (built
once per model, cached next to the panel cache); any wind direction is then
a cheap product. Speeds are relative to the wind (1 = wind speed).

Each line also carries a "flow time" (distance / local speed), so dashes
moving along it at a steady rate in flow time speed up and slow down with
the air, which is what makes it read as moving air.

Potential flow has no wake, so behind the model the lines close up again
instead of swirling.
'''

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aero import colors, panel
from aero.geometry import panel_rings

FIELD_VERSION = 2                # 2: also the wings' circulation (G_lift)
GRID = 128                 # ground grid cells per side
DOMAIN = 2.3               # ground grid half-size, in model radii (footprint centre to farthest corner)
MIN_SPEED = 0.03           # a line stops where the air is this slow (stagnation), x wind speed
RING_MAX = 24              # candidate triangles per step (the nearest that share a corner)
TIME_SPEED_FLOOR = 0.1     # flow time uses at least this speed, so a near-stop doesn't freeze the dashes

# Line speed colours, slow -> fast: dim violet, magenta, orange, pale yellow.
# Drawn additively over the model, so slow air is faint and fast air bright.
SPEED_STOPS = np.array([
    [0.10, 0.02, 0.25],
    [0.45, 0.05, 0.55],
    [0.85, 0.20, 0.45],
    [0.98, 0.55, 0.15],
    [1.00, 0.95, 0.65],
])


def model_size(panels):
    return float(np.ptp(panels.vertices, axis=0).max())


@dataclass
class LineSet:
    """Polylines, concatenated. Line k is points[starts[k]:starts[k + 1]]."""
    points: np.ndarray      # (P, 3) plate frame, mm, already lifted off the surface
    normals: np.ndarray     # (P, 3) the surface each point lies on (ribbons lie flat on it)
    speed: np.ndarray       # (P,) x wind speed
    time: np.ndarray        # (P,) flow time along the line (mm at wind speed)
    starts: np.ndarray      # (L + 1,)
    ground: np.ndarray      # (L,) True for ground lines
    yaw_deg: float = 0.0

    def __len__(self):
        return len(self.starts) - 1


def pack_lines(lines, yaw):
    """[(points, normals, speed, ground)] -> LineSet, with flow time added."""
    pts, nrm, spd, tim, starts, ground = [], [], [], [], [0], []
    for p, n, q, g in lines:
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        q_mid = 0.5 * (q[1:] + q[:-1])
        tim.append(np.concatenate([[0.0], np.cumsum(step / np.maximum(q_mid, TIME_SPEED_FLOOR))]))
        pts.append(p), nrm.append(n), spd.append(q), ground.append(g)
        starts.append(starts[-1] + len(p))
    if not pts:
        empty = np.zeros((0, 3))
        return LineSet(empty, empty, np.zeros(0), np.zeros(0), np.zeros(1, int), np.zeros(0, bool), yaw)
    return LineSet(np.concatenate(pts), np.concatenate(nrm), np.concatenate(spd), np.concatenate(tim),
                   np.array(starts), np.array(ground, bool), yaw)


def resample(points, spacing, *values):
    """Polyline -> evenly spaced points (values interpolated alongside)."""
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(step)])
    if s[-1] <= spacing:
        return (points, *values)
    t = np.linspace(0.0, s[-1], int(np.ceil(s[-1] / spacing)) + 1)
    out = [np.stack([np.interp(t, s, points[:, k]) for k in range(points.shape[1])], axis=1)]
    for v in values:
        v = np.asarray(v, np.float64)
        out.append(np.interp(t, s, v) if v.ndim == 1 else
                   np.stack([np.interp(t, s, v[:, k]) for k in range(v.shape[1])], axis=1))
    return tuple(out)


# ---------------------------------------------------------------- ground lines

class GroundField:
    """Velocity matrices on a grid just above the plate, and which cells are
    under the model. G_lift: the same with the wings' circulation (models with
    wings), blended in by panel.lift_weight() like the surface."""

    def __init__(self, G, inside, radius, height, key="", G_lift=None):
        self.G = np.asarray(G, np.float64)          # (n, n, 2, 2), [iy, ix]
        self.G_lift = None if G_lift is None else np.asarray(G_lift, np.float64)
        self.inside = np.asarray(inside, bool)      # (n, n)
        self.radius = float(radius)                 # grid covers [-radius, radius]^2
        self.height = float(height)                 # where it was sampled, mm above the plate
        self.key = key

    @property
    def n(self):
        return self.G.shape[0]

    @property
    def cell(self):
        return 2.0 * self.radius / (self.n - 1)

    def velocity(self, wind):
        """(n, n, 2) velocity / wind speed on the grid for a unit wind along the plate."""
        V = np.einsum("yxij,j->yxi", self.G, wind[:2])
        w = float(panel.lift_weight(wind)) if self.G_lift is not None else 0.0
        if w > 0:
            V = (1.0 - w) * V + w * np.einsum("yxij,j->yxi", self.G_lift, wind[:2])
        return V

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        extra = {} if self.G_lift is None else {"G_lift": self.G_lift.astype(np.float32)}
        np.savez_compressed(path, G=self.G.astype(np.float32), inside=self.inside, radius=self.radius,
                            height=self.height, key=self.key, version=FIELD_VERSION, **extra)

    @classmethod
    def load(cls, path):
        with np.load(path) as d:
            return cls(d["G"], d["inside"], float(d["radius"]), float(d["height"]), str(d["key"]),
                       d["G_lift"] if "G_lift" in d.files else None)


def _under_model(panels, xy, height):
    """True where a point on the plate is covered by the model: a ray straight
    up from it crosses the surface an odd number of times."""
    tri = panels.triangles
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    use = np.abs(det) > 1e-12 * np.abs(det).max()        # vertical walls never cross a vertical ray
    a, b, c, det, z = a[use], b[use], c[use], det[use], tri[use, :, 2]
    crossings = np.zeros(len(xy), int)
    for start in range(0, len(xy), 512):
        p = xy[start:start + 512, None, :]
        u = ((b[:, 0] - p[..., 0]) * (c[:, 1] - p[..., 1]) - (c[:, 0] - p[..., 0]) * (b[:, 1] - p[..., 1])) / det
        v = ((c[:, 0] - p[..., 0]) * (a[:, 1] - p[..., 1]) - (a[:, 0] - p[..., 0]) * (c[:, 1] - p[..., 1])) / det
        w = 1.0 - u - v
        hit = (u >= 0) & (v >= 0) & (w >= 0)
        z_hit = u * z[:, 0] + v * z[:, 1] + w * z[:, 2]
        crossings[start:start + 512] = np.sum(hit & (z_hit > height), axis=1)
    return crossings % 2 == 1


def build_ground_field(panels, solution, n=GRID, progress=None, cancel=None):
    xy_extent = np.abs(panels.vertices[:, :2]).max(axis=0)
    radius = DOMAIN * float(np.hypot(*xy_extent))
    height = 0.002 * model_size(panels)
    axis = np.linspace(-radius, radius, n)
    X, Y = np.meshgrid(axis, axis)                      # [iy, ix]
    xy = np.column_stack([X.ravel(), Y.ravel()])
    inside = _under_model(panels, xy, height)
    G = np.tile(np.eye(3), (len(xy), 1, 1))
    G_lift = G.copy() if solution.lifting is not None else None
    outside = np.nonzero(~inside)[0]
    points = np.column_stack([xy[outside], np.full(len(outside), height)])
    G[outside], lifted = panel.velocity_matrices(panels, solution, points, progress=progress, cancel=cancel)
    if G_lift is not None:
        G_lift[outside] = lifted
    # Grow the footprint by a cell, so lines stop just short of the model's foot.
    mask = inside.reshape(n, n)
    grown = mask.copy()
    grown[1:] |= mask[:-1]; grown[:-1] |= mask[1:]; grown[:, 1:] |= mask[:, :-1]; grown[:, :-1] |= mask[:, 1:]
    return GroundField(G[:, :2, :2].reshape(n, n, 2, 2), grown, radius, height, panels.key,
                       None if G_lift is None else G_lift[:, :2, :2].reshape(n, n, 2, 2))


def field_file(info, panels):
    return info.aero_dir / f"field_{panels.key}_p{panel.SOLVER_VERSION}_v{FIELD_VERSION}.npz"


def load_or_build_field(info, panels, solution, progress=None, cancel=None):
    path = field_file(info, panels)
    if path.exists():
        return GroundField.load(path)
    field = build_ground_field(panels, solution, progress=progress, cancel=cancel)
    field.save(path)
    return field


def max_line_counts(panels, field, width):
    """How many lines fit before they run into each other: ground lines at
    least 4 widths apart across the rake, surface seeds about 6 widths apart.
    Small models (a 20 mm cube with 1.5 mm lines) get far fewer."""
    wetted = float(panels.areas.sum())
    rake = 2.0 * field.radius / np.sqrt(2.0)
    return max(1, int(wetted / (6.0 * width) ** 2)), max(1, int(rake / (4.0 * width)))


def ground_lines(field, yaw_deg, count=28, lift=0.0, spacing=None):
    """Streamlines over the plate for this wind, as [(points, normals, speed, True)].

    The lines run through a square aligned with the wind, centred on the
    model, so the picture turns with the wind (the grid is a circle's worth
    bigger). Seeds on its upstream edge; a line ends where it leaves the
    square, reaches the model, or the air stops. Lines that end at the model
    leave a gap behind it, so the same rake on the downstream edge is traced
    backwards, and those of its lines that end at the model fill the gap."""
    spacing = max(spacing or 0.0, 0.4 * field.cell)
    out = []
    for direction in (1.0, -1.0):
        for xy, q, reached_far_edge in _rake(field, yaw_deg, count, direction, step=max(0.6 * field.cell, 0.5 * spacing)):
            if direction < 0:
                if reached_far_edge:
                    continue                # the same line as one from upstream
                xy, q = xy[::-1], q[::-1]
            pts = np.column_stack([xy, np.full(len(xy), field.height + lift)])
            pts, q = resample(pts, spacing, q)
            out.append((pts, np.tile([0.0, 0.0, 1.0], (len(pts), 1)), q, True))
    return out


def _rake(field, yaw_deg, count, direction, step):
    """Trace `count` lines from the upstream edge (direction 1) or backwards
    from the downstream edge (-1). Returns [(xy, speed, reached the other edge)]."""
    a = np.radians(yaw_deg)
    wind, across = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    V = field.velocity(np.array([wind[0], wind[1], 0.0]))  # (n, n, 2) velocity / wind speed
    half = field.radius / np.sqrt(2.0)
    cell, n = field.cell, field.n

    def sample(p):
        f = (p + field.radius) / cell
        i = np.clip(np.floor(f).astype(int), 0, n - 2)
        t = np.clip(f - i, 0.0, 1.0)
        x0, y0, tx, ty = i[:, 0], i[:, 1], t[:, :1], t[:, 1:]
        v = ((1 - tx) * (1 - ty) * V[y0, x0] + tx * (1 - ty) * V[y0, x0 + 1]
             + (1 - tx) * ty * V[y0 + 1, x0] + tx * ty * V[y0 + 1, x0 + 1])
        blocked = field.inside[np.clip(np.rint(f[:, 1]).astype(int), 0, n - 1),
                               np.clip(np.rint(f[:, 0]).astype(int), 0, n - 1)]
        return v, blocked

    s = np.linspace(-half, half, count + 2)[1:-1]
    p = -direction * half * wind + s[:, None] * across
    h = step * direction
    alive = np.ones(count, bool)
    far_edge = np.zeros(count, bool)
    track, speeds, live = [p.copy()], [np.linalg.norm(sample(p)[0], axis=1)], [alive.copy()]
    for _ in range(int(4 * half / abs(h))):
        v, _ = sample(p)
        mid = p + 0.5 * h * v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        v_mid, _ = sample(mid)
        speed = np.linalg.norm(v_mid, axis=1)
        p = np.where(alive[:, None], p + h * v_mid / np.maximum(speed, 1e-9)[:, None], p)
        _, blocked = sample(p)
        along, side = direction * (p @ wind), p @ across
        far_edge |= alive & (along >= half)
        alive &= (speed > MIN_SPEED) & ~blocked & (along < half) & (np.abs(side) < half)
        track.append(p.copy()), speeds.append(speed), live.append(alive.copy())
        if not alive.any():
            break

    track, speeds, live = np.array(track), np.array(speeds), np.array(live)
    out = []
    for k in range(count):
        m = int(live[:, k].sum()) + 1
        if m >= 3:
            out.append((track[:m, k], speeds[:m, k], bool(far_edge[k])))
    return out


# --------------------------------------------------------------- surface lines

def _closest_on_triangles(p, a, b, c):
    """Closest point to p on each triangle (Ericson's region tests), vectorised."""
    ab, ac, ap = b - a, c - a, p - a
    bp, cp = p - b, p - c
    dot = lambda x, y: np.einsum("...k,...k->...", x, y)
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 / (va + vb + vc)
        out = a + ab * (vb * denom)[..., None] + ac * (vc * denom)[..., None]
        # Lowest priority first, so the earlier regions in Ericson's order win.
        bc_t = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        out = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * bc_t[..., None], out)
        out = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * (d2 / (d2 - d6))[..., None], out)
        out = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)
        out = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * (d1 / (d1 - d3))[..., None], out)
        out = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)
        out = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)
    return out


def farthest_points(points, count):
    """Indices of `count` points spread out evenly (greedy farthest-point sampling)."""
    count = min(count, len(points))
    chosen = [int(np.argmin(points[:, 2] - 1e-3 * points[:, 0]))]
    d = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(count - 1):
        chosen.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.array(chosen)


class SurfaceTracer:
    """Walks lines over the model's triangles. Built once per model."""

    def __init__(self, panels):
        self.panels = panels
        self.tri = panels.triangles
        self.normals = panels.normals
        self.areas = panels.areas
        self.size = model_size(panels)
        self.step_raw = 0.5 * np.sqrt(2.0 * self.areas)      # about half a triangle
        self.rings = panel_rings(panels, RING_MAX)

    def _project(self, p, tri):
        cand = self.rings[tri]                                          # (L, K)
        t = self.tri[cand]                                              # (L, K, 3, 3)
        q = _closest_on_triangles(p[:, None, :], t[:, :, 0], t[:, :, 1], t[:, :, 2])
        d = np.linalg.norm(q - p[:, None, :], axis=2)
        best = np.argmin(d, axis=1)
        rows = np.arange(len(p))
        return q[rows, best], cand[rows, best], d[rows, best]

    def _velocity(self, p, tri, vertex_velocity):
        a, b, c = self.tri[tri, 0], self.tri[tri, 1], self.tri[tri, 2]
        v0, v1, v2 = b - a, c - a, p - a
        dot = lambda x, y: np.einsum("ik,ik->i", x, y)
        d00, d01, d11, d20, d21 = dot(v0, v0), dot(v0, v1), dot(v1, v1), dot(v2, v0), dot(v2, v1)
        denom = np.maximum(d00 * d11 - d01 * d01, 1e-300)
        wb = np.clip((d11 * d20 - d01 * d21) / denom, 0, 1)
        wc = np.clip((d00 * d21 - d01 * d20) / denom, 0, 1)
        wa = np.clip(1.0 - wb - wc, 0, 1)
        f = self.panels.faces[tri]
        v = (wa[:, None] * vertex_velocity[f[:, 0]] + wb[:, None] * vertex_velocity[f[:, 1]]
             + wc[:, None] * vertex_velocity[f[:, 2]])
        n = self.normals[tri]
        return v - np.einsum("ik,ik->i", v, n)[:, None] * n           # keep it on this face

    def _trace(self, start, tri, vertex_velocity, direction, max_length, step):
        """Step every line until it stops. Returns per-line point and triangle lists."""
        p, t = start.copy(), tri.copy()
        active = np.arange(len(p))                  # lines still going (only these are stepped)
        length = np.zeros(len(p))
        floor = 0.01 * self.size
        points = [[x] for x in p]
        tris = [[x] for x in t]
        for _ in range(int(1.2 * max_length / step.min())):
            v = direction * self._velocity(p, t, vertex_velocity)
            speed = np.linalg.norm(v, axis=1)
            h = step[t]
            moved = p + h[:, None] * v / np.maximum(speed, 1e-12)[:, None]
            q, t_new, d = self._project(moved, t)
            travelled = np.linalg.norm(q - p, axis=1)
            ok = (speed > MIN_SPEED) & (d < 0.6 * h) & (q[:, 2] > floor) & (travelled > 0.05 * h)
            length[active] += np.where(ok, travelled, 0.0)
            ok &= length[active] < max_length
            for i, line in enumerate(active[ok]):
                points[line].append(q[ok][i])
                tris[line].append(t_new[ok][i])
            active, p, t = active[ok], q[ok], t_new[ok]
            if len(active) == 0:
                break
        return [np.array(x) for x in points], [np.array(x) for x in tris]

    def lines(self, panel_velocity, count=60, lift=0.0, spacing=None):
        """Surface streamlines for this flow, as [(points, normals, speed, False)].
        panel_velocity: (N, 3) surface velocity / wind speed on each panel."""
        vertex_velocity = np.stack([self.panels.to_vertices(panel_velocity[:, k]) for k in range(3)], axis=1)
        seeds = farthest_points(self.panels.centroids, count)
        start = self.panels.centroids[seeds]
        spacing = spacing or 0.004 * self.size
        # Steps finer than half the output spacing only cost time.
        step = np.clip(self.step_raw, max(0.005 * self.size, 0.5 * spacing), max(0.02 * self.size, spacing))
        max_length = 1.5 * self.size
        fwd_pts, fwd_tris = self._trace(start, seeds, vertex_velocity, 1.0, max_length, step)
        back_pts, back_tris = self._trace(start, seeds, vertex_velocity, -1.0, max_length, step)
        out = []
        for k in range(len(seeds)):
            # Upstream half reversed, then downstream (the seed point once).
            pts = np.concatenate([back_pts[k][::-1], fwd_pts[k][1:]])
            ts = np.concatenate([back_tris[k][::-1], fwd_tris[k][1:]])
            if len(pts) < 3 or np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() < 0.03 * self.size:
                continue
            q = np.linalg.norm(panel_velocity[ts], axis=1)
            nrm = self.normals[ts]
            pts, q, nrm = resample(pts, spacing, q, nrm)
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
            out.append((pts + lift * nrm, nrm, q, False))
        return out


# --------------------------------------------------------------------- ribbons

@dataclass
class Ribbons:
    """Flat strips along the lines: two points per line point, two triangles per segment."""
    points: np.ndarray      # (V, 3)
    triangles: np.ndarray   # (T, 3) uint32
    speed: np.ndarray       # (V,)
    time: np.ndarray        # (V,)


def ribbons(lines, width):
    """Strips `width` mm wide lying on the surface each line is on."""
    if len(lines) == 0:
        return Ribbons(np.zeros((0, 3)), np.zeros((0, 3), np.uint32), np.zeros(0), np.zeros(0))
    P, N = lines.points, lines.normals
    tangent = np.zeros_like(P)
    for k in range(len(lines)):
        s, e = lines.starts[k], lines.starts[k + 1]
        tangent[s:e] = np.gradient(P[s:e], axis=0) if e - s > 1 else 0
    side = np.cross(N, tangent)
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-12)
    points = np.empty((2 * len(P), 3))
    points[0::2] = P - 0.5 * width * side
    points[1::2] = P + 0.5 * width * side

    tris = []
    for k in range(len(lines)):
        s, e = lines.starts[k], lines.starts[k + 1]
        i = 2 * np.arange(s, e - 1)
        tris.append(np.column_stack([i, i + 1, i + 2]))
        tris.append(np.column_stack([i + 1, i + 3, i + 2]))
    return Ribbons(points, np.concatenate(tris).astype(np.uint32),
                   np.repeat(lines.speed, 2), np.repeat(lines.time, 2))


def ribbon_colors(rib, seconds=0.0, rate=0.0, dash=1.0, low=0.0, high=2.0):
    """(V, 4) RGBA, colour already multiplied by alpha. rate: how far the
    dashes move per second (flow time, mm); rate 0 draws solid lines. Each
    dash is opaque and bright at its head and fades back to see-through, so
    the skin shows between dashes (and the plate stays dark)."""
    rgb = colors.colormap(rib.speed, low, high, SPEED_STOPS)
    alpha = np.ones(len(rgb))
    if rate > 0:
        phase = np.mod((rib.time - seconds * rate) / dash, 1.0)
        alpha = 0.12 + 0.88 * phase ** 3
    return np.column_stack([rgb * alpha[:, None], alpha]).astype(np.float32)
