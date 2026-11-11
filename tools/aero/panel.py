'''panel.py: pipeline 1, the traditional solver. A 3D source panel method.

Imported by: aero/pipeline.py, aero/precompute.py, aero/train.py, tests
Run directly: no (python -m aero.precompute builds the cache from the command line)

Potential flow (inviscid, incompressible) around the model, with the plate as
the ground via a mirror image. Each wetted triangle carries a constant source
strength; strengths are chosen so no air goes through any panel centre
(Hess & Smith). The surface speed gives the pressure coefficient
Cp = 1 - (V / V_inf)^2.

The heavy part does not depend on the wind. The equations are linear in the
wind, so once per model this builds, for every panel i, a 3x3 matrix M_i with
    V_i = M_i @ V_inf
After that any wind direction and speed costs one small matrix product, so
this pipeline also runs live. The matrices are cached in
models/<name>/aero/, keyed by the geometry, so the build (a few seconds to a
minute) only happens once per model.

What potential flow cannot show: separation and the wake. The front and the
suction over the top are right; the low pressure it predicts behind sharp
edges and on the downwind side is too strong. To get those right, train the
network on CFD results (train.py --data).

Wings: sources alone cannot make lift. A model with trailing edges (sharp
back edges of wings, found by lifting.trailing_edges) is solved by lifting.py
instead: doublets, the Kutta condition and a wake, still linear in the wind,
so still one 3x3 matrix per panel. It keeps two solutions, with and without
the wake, and lift_weight() blends between them by the wind's angle: the
fixed wake is right while the wind comes from ahead, not from the side.
A model turns it on with "kutta": true in its model.json. The tests check it
on clean wings (lift slope within 4% of lifting-line theory, ground effect).
On printed models whose wings end in rounded edges or were decimated
coarsely it can come out wrong (see models/README.md), so it is off by default.
'''

from pathlib import Path

import numpy as np

# 2: also keeps the source strengths, for the velocity off the surface (flowlines.py).
# 3: models with trailing edges are solved with lift (lifting.py).
SOLVER_VERSION = 3

# The wake trails straight back along the model's +X, which is only right for
# wind from ahead: lift counts fully within this angle of the nose, and fades
# out by the second (smoothly in between).
LIFT_FULL_DEG, LIFT_NONE_DEG = 40.0, 75.0

# Within this many panel sizes the exact triangle formula is used; further
# away a triangle looks like a point source (error well under 1%).
NEAR_FIELD = 5.0
ROWS_PER_CHUNK = 128

_FOUR_PI = 4.0 * np.pi


def solid_angle(R, r):
    """Signed solid angle each triangle covers, seen from the point: R (M, 3, 3)
    point -> corners, r (M, 3) their lengths (van Oosterom & Strackee).
    Positive on the triangle's outside (the side its normal points to)."""
    triple = np.einsum("ij,ij->i", R[:, 0], np.cross(R[:, 1], R[:, 2]))
    denom = (r[:, 0] * r[:, 1] * r[:, 2]
             + np.einsum("ij,ij->i", R[:, 0], R[:, 1]) * r[:, 2]
             + np.einsum("ij,ij->i", R[:, 0], R[:, 2]) * r[:, 1]
             + np.einsum("ij,ij->i", R[:, 1], R[:, 2]) * r[:, 0])
    return -2.0 * np.arctan2(triple, denom)


def source_velocity(points, triangles, self_term=None):
    """Velocity at each point from a unit-strength source spread over the
    paired triangle (points (M, 3), triangles (M, 3, 3), corners counter-clockwise
    seen from outside). self_term: bool (M,), points on their own triangle,
    which get the outside limit (half the source leaves through each side)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area_normal = np.cross(b - a, c - a)
    n = area_normal / np.linalg.norm(area_normal, axis=1, keepdims=True)

    R = triangles - points[:, None, :]                  # point -> each corner
    r = np.linalg.norm(R, axis=2)

    # Normal part: the solid angle the triangle covers, seen from the point.
    omega = solid_angle(R, r)
    if self_term is not None:
        omega = np.where(self_term, 2.0 * np.pi, omega)

    # In-plane part: by the divergence theorem, a sum over the edges of the
    # edge's outward normal times the integral of 1/distance along it.
    tangential = np.zeros_like(points)
    for k in range(3):
        start, end = triangles[:, k], triangles[:, (k + 1) % 3]
        edge = end - start
        length = np.linalg.norm(edge, axis=1)
        outward = np.cross(edge / length[:, None], n)
        ra, rb = r[:, k], r[:, (k + 1) % 3]
        log_term = np.log((ra + rb + length) / np.maximum(ra + rb - length, 1e-12 * length))
        tangential += outward * log_term[:, None]

    return (tangential + omega[:, None] * n) / _FOUR_PI


def mirror(triangles):
    """The image below the plate (Z -> -Z), corners reordered so it still faces out."""
    image = triangles[:, [0, 2, 1]].copy()
    image[..., 2] *= -1.0
    return image


class _Influence:
    """Builds rows of B, where B[i, j] is the velocity at panel i's centre from
    a unit source on panel j (and on its mirror image, which has the same
    strength because the flow is symmetric about the plate)."""

    def __init__(self, panels, ground):
        self.n = len(panels)
        self.points = panels.centroids
        tri = panels.triangles
        self.sources = np.concatenate([tri, mirror(tri)]) if ground else tri
        self.centres = self.sources.mean(axis=1)
        self.areas = 0.5 * np.linalg.norm(np.cross(self.sources[:, 1] - self.sources[:, 0],
                                                   self.sources[:, 2] - self.sources[:, 0]), axis=1)
        edges = np.stack([self.sources[:, (k + 1) % 3] - self.sources[:, k] for k in range(3)], axis=1)
        self.near = NEAR_FIELD * np.linalg.norm(edges, axis=2).max(axis=1)

    def rows(self, start, stop):
        return self.at(self.points[start:stop], own=np.arange(start, stop))

    def at(self, points, own=None):
        """B for any points (c, 3) -> (c, n, 3). own[i] is the panel point i
        sits on, if it is a panel centre."""
        offset = points[:, None, :] - self.centres[None, :, :]          # (c, J, 3)
        distance = np.linalg.norm(offset, axis=2)
        # A panel's own centre is at distance 0; that pair is always near and
        # gets the exact formula below.
        safe = np.maximum(distance, 1e-9 * self.near.min())
        far = self.areas[None, :, None] * offset / (_FOUR_PI * safe[..., None] ** 3)

        rows, cols = np.nonzero(distance < self.near[None, :])
        far[rows, cols] = source_velocity(points[rows], self.sources[cols],
                                          self_term=None if own is None else (cols == own[rows]))
        if len(self.sources) == self.n:
            return far
        return far[:, :self.n] + far[:, self.n:]


def lift_weight(wind):
    """How much of the lifting solution to use for this wind (0..1): all of it
    while the wind comes from within LIFT_FULL_DEG of the nose (along +X),
    none past LIFT_NONE_DEG, smoothly in between."""
    wind = np.asarray(wind, np.float64)
    along = np.hypot(wind[..., 0], wind[..., 1])
    cos_angle = np.where(along > 1e-12, wind[..., 0] / np.maximum(along, 1e-12), 1.0)
    full, none = np.cos(np.radians(LIFT_FULL_DEG)), np.cos(np.radians(LIFT_NONE_DEG))
    t = np.clip((cos_angle - none) / (full - none), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class PanelSolution:
    """The per-panel 3x3 matrices that turn a wind vector into surface velocity,
    and the source strengths per unit wind (S, for the flow off the surface).
    A model with wings also has M_lift (the same with the wake's circulation)
    and lifting.LiftingParts (doublets and wake, for the flow off the surface)."""

    def __init__(self, M, key="", S=None, M_lift=None, lifting=None):
        self.M = np.asarray(M, np.float64)        # (N, 3, 3)
        self.key = key
        self.S = None if S is None else np.asarray(S, np.float64)   # (N, 3): sigma = S @ wind
        self.M_lift = None if M_lift is None else np.asarray(M_lift, np.float64)
        self.lifting = lifting

    def surface_velocity(self, wind):
        """(N, 3) surface velocity for a wind vector (any length)."""
        wind = np.asarray(wind, np.float64)
        v = self.M @ wind
        if self.M_lift is not None:
            w = float(lift_weight(wind))
            if w > 0:
                v = (1.0 - w) * v + w * (self.M_lift @ wind)
        return v

    def cp(self, wind):
        """Pressure coefficient on each panel. Only the direction of `wind`
        matters: potential flow scales with speed, so Cp does not."""
        wind = np.asarray(wind, np.float64)
        v = self.surface_velocity(wind / np.linalg.norm(wind))
        return 1.0 - np.einsum("ij,ij->i", v, v)

    def cp_batch(self, winds):
        """(S, N) Cp for S unit wind vectors."""
        winds = np.asarray(winds, np.float64)
        winds = winds / np.linalg.norm(winds, axis=1, keepdims=True)
        v = np.einsum("nij,sj->sni", self.M, winds)
        if self.M_lift is not None:
            w = lift_weight(winds)[:, None, None]
            v = (1.0 - w) * v + w * np.einsum("nij,sj->sni", self.M_lift, winds)
        return 1.0 - np.einsum("sni,sni->sn", v, v)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {}
        if self.M_lift is not None:
            parts = self.lifting
            extra = dict(M_lift=self.M_lift.astype(np.float32), mu=parts.mu, mu_lift=parts.mu_lift,
                         wake=parts.wake, upper=parts.upper, lower=parts.lower)
        np.savez_compressed(path, M=self.M.astype(np.float32), S=self.S, key=self.key, version=SOLVER_VERSION,
                            **extra)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            S = data["S"] if "S" in data.files and data["S"].ndim == 2 else None
            if "M_lift" not in data.files:
                return cls(data["M"], str(data["key"]), S)
            from aero.lifting import LiftingParts
            parts = LiftingParts(data["mu"], data["mu_lift"], data["wake"], data["upper"], data["lower"])
            return cls(data["M"], str(data["key"]), S, data["M_lift"], parts)


class Cancelled(Exception):
    pass


def build(panels, ground=True, progress=None, cancel=None, lift=False):
    """Solve for the panel matrices. progress(fraction) is called as it goes;
    cancel() returning True stops it (raises Cancelled). With lift, a model
    with trailing edges is solved with lifting.py."""
    if lift:
        from aero import lifting
        from aero.geometry import panel_rings, thin_partners
        corners, upper, lower = lifting.trailing_edges(panels, thin_partners(panels))
        if len(corners):
            rings = panel_rings(panels, lifting.GRADIENT_RING)
            def tick(fraction):
                if cancel is not None and cancel():
                    raise Cancelled()
                if progress is not None:
                    progress(fraction)
            M, M_lift, parts = lifting.build(panels, corners, upper, lower, rings, ground, tick)
            return PanelSolution(M, panels.key, -panels.normals, M_lift, parts)

    influence = _Influence(panels, ground)
    n = len(panels)
    normals = panels.normals
    chunks = [(s, min(s + ROWS_PER_CHUNK, n)) for s in range(0, n, ROWS_PER_CHUNK)]
    steps = 2 * len(chunks)

    def tick(done):
        if cancel is not None and cancel():
            raise Cancelled()
        if progress is not None:
            progress(done / steps)

    # Pass 1: the flow-tangency equations, A @ sigma = -normals @ wind.
    A = np.empty((n, n))
    for step, (start, stop) in enumerate(chunks):
        A[start:stop] = np.einsum("cjk,ck->cj", influence.rows(start, stop), normals[start:stop])
        tick(step + 1)
    # One solve for all three wind components: sigma = S @ wind.
    S = np.linalg.solve(A, -normals)
    del A

    # Pass 2: surface velocity V_i = wind + sum_j B_ij sigma_j = M_i @ wind.
    # B is rebuilt rather than kept, so memory stays at one n x n matrix.
    M = np.empty((n, 3, 3))
    for step, (start, stop) in enumerate(chunks):
        M[start:stop] = np.eye(3) + np.einsum("cjk,jl->ckl", influence.rows(start, stop), S)
        tick(len(chunks) + step + 1)
    return PanelSolution(M, panels.key, S)


def velocity_matrices(panels, solution, points, ground=True, chunk=256, progress=None, cancel=None):
    """(G, G_lift): (c, 3, 3) matrices with velocity = G @ wind at any points
    off the surface: the wind plus what every source (and its mirror) adds
    there. G_lift is the same with the wings' circulation (None without wings);
    blend them with lift_weight() as surface_velocity() does."""
    if solution.lifting is not None:
        from aero import lifting
        return lifting.velocity_matrices(panels, solution.lifting, points, ground, progress=progress, cancel=cancel)
    influence = _Influence(panels, ground)
    points = np.asarray(points, np.float64)
    G = np.empty((len(points), 3, 3))
    for start in range(0, len(points), chunk):
        if cancel is not None and cancel():
            raise Cancelled()
        stop = min(start + chunk, len(points))
        G[start:stop] = np.eye(3) + np.einsum("cjk,jl->ckl", influence.at(points[start:stop]), solution.S)
        if progress is not None:
            progress(stop / len(points))
    return G, None


def cache_file(info, panels):
    lift = "_lift" if info.kutta else ""
    return info.aero_dir / f"panel_{panels.key}_v{SOLVER_VERSION}{lift}.npz"


def load_or_build(info, panels, progress=None, cancel=None, save=True):
    """The cached solution for this model and geometry, built if missing."""
    path = cache_file(info, panels)
    if path.exists():
        return PanelSolution.load(path)
    solution = build(panels, progress=progress, cancel=cancel, lift=info.kutta)
    if save:
        solution.save(path)
    return solution
