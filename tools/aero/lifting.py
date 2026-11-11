'''lifting.py: the panel method for models with wings. Doublets, the Kutta
condition and a wake, so wings make lift (on an F1 car: downforce).

Imported by: aero/panel.py (which uses it for models with trailing edges), tests
Run directly: no

The plain solver (panel.py) uses only sources, and sources cannot make lift:
lift needs circulation. This is the standard way to add it (Morino's
formulation, as in Katz & Plotkin, "Low-Speed Aerodynamics", ch. 12):

  - every panel carries a source of known strength, sigma = -n . wind, and a
    doublet of unknown strength mu;
  - the air inside the body is kept still: the potential just inside every
    panel is zero (the Dirichlet condition). That is one equation per panel;
  - at each trailing edge (the sharp back edge of a wing) the air leaves
    smoothly: the Kutta condition. A wake of doublet panels trails straight
    back from the edge, with mu_wake = mu_upper - mu_lower;
  - the surface speed is the wind along the surface plus the gradient of mu
    along it, which gives Cp as before.

The wake trails along the model's own +X (the downstream direction at home),
so the equations stay linear in the wind, and the solution is still a 3x3
matrix per panel: any wind costs one small matrix product. A wake fixed to the
model is right when the wind comes from ahead; panel.lift_weight() blends this
lifting solution into the plain one (the same equations without the wake) as
the wind swings round to the side.

The plate is the ground: sources, doublets and wakes all have mirror images
below it, with the same strengths, and those images give the ground effect.

Still potential flow: no stall, no separation, no viscous wake, and on a
10 cm model the real air is more viscous than this assumes, so lift comes out
too high. The pattern (suction under an inverted wing, more of it near the
ground, air speeding up and turning round the wings) is what it gets right.
'''

import numpy as np

from aero.panel import NEAR_FIELD, mirror, solid_angle, source_velocity

STREAM = np.array([1.0, 0.0, 0.0])  # the model's downstream direction: wakes trail this way
WAKE_LENGTH = 20.0                   # wake length, in model sizes (long enough to look infinite)
WAKE_START = 0.05                    # mm: look for obstacles from just behind the trailing edge
WAKE_CLEARANCE = 0.5                 # mm: a wake that would run into the model stops this short of it
# No wing carries more circulation than this, per unit wind, as a fraction of
# the model size (it is already a lift coefficient beyond 3 on a wing spanning
# the model). A strip that comes out stronger is a numerical failure (a
# rounded edge on a coarse mesh, two wakes almost on top of each other): it
# is dropped and the rest solved again, so that part makes no lift.
GAMMA_LIMIT = 0.5

# Trailing edges are where a wing's top surface ends at the back. A wing's
# top surface: panels on a thin part (geometry.thin_partners: a panel on the
# other side within a wing's thickness) facing mostly up (TE_UP). Its back
# edge: the edges where that region ends, running across the stream
# (TE_ACROSS), with the region's outside facing downstream (TE_BACK). This
# finds sharp, blunt (flat back face) and rounded trailing edges alike, which
# matters for printable models; square-backed solids such as the cube have no
# thin parts, so no trailing edges.
TE_UP = 0.5
TE_ACROSS = 0.7
TE_BACK = 0.5

# Wheels shed no wake: a part whose tread (panels facing across the axle,
# |n . axle| < WHEEL_TREAD) covers at least WHEEL_AREA of it and whose tread
# normals all point away from one horizontal line (the axle, within
# WHEEL_ROUND of the radius) is round, not a wing.
WHEEL_TREAD = 0.3
WHEEL_AREA = 0.3
WHEEL_ROUND = 0.15

GRADIENT_RING = 24                   # neighbours used for the surface gradient of mu
GRADIENT_SPREAD = 0.02               # neighbours must spread this well round a panel to fit its slope
SAME_SURFACE = 0.5                   # neighbours across a sharper turn than this are another surface
ROWS_PER_CHUNK = 64

_FOUR_PI = 4.0 * np.pi


# ------------------------------------------------------------------ kernels

def potentials(points, triangles, self_term=None):
    """Potential at each point from a unit source and from a unit doublet
    spread over the paired triangle (corners counter-clockwise seen from
    outside). Returns (phi_source, phi_doublet), each (M,).

    Source: -1/(4 pi) * integral of dA / R, whose gradient is source_velocity().
    Doublet: solid angle / (4 pi), which jumps by 1 across the triangle,
    from -1/2 just inside to +1/2 just outside. self_term: points at their own
    triangle's centre, evaluated just inside (the Dirichlet condition)."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    area_normal = np.cross(b - a, c - a)
    n = area_normal / np.linalg.norm(area_normal, axis=1, keepdims=True)
    R = triangles - points[:, None, :]
    r = np.linalg.norm(R, axis=2)
    omega = solid_angle(R, r)

    # integral of dA / R over a flat polygon: a sum over the edges of the
    # in-plane distance to the edge times the integral of 1/R along it, less
    # height * solid angle.
    height = np.einsum("ij,ij->i", points - a, n)
    integral = -height * omega
    for k in range(3):
        start, end = triangles[:, k], triangles[:, (k + 1) % 3]
        edge = end - start
        length = np.linalg.norm(edge, axis=1)
        outward = np.cross(edge / length[:, None], n)
        ra, rb = r[:, k], r[:, (k + 1) % 3]
        log_term = np.log((ra + rb + length) / np.maximum(ra + rb - length, 1e-12 * length))
        integral += np.einsum("ij,ij->i", start - points, outward) * log_term

    phi_doublet = omega / _FOUR_PI
    if self_term is not None:
        phi_doublet = np.where(self_term, -0.5, phi_doublet)
    return -integral / _FOUR_PI, phi_doublet


def doublet_velocity(points, triangles):
    """Velocity at each point from a unit doublet on the paired triangle: the
    gradient of its potential, which is a vortex ring round the triangle's
    edges (clockwise seen from outside, so the flow through it runs inward)."""
    u = np.zeros_like(points)
    for k in range(3):
        start, end = triangles[:, k], triangles[:, (k + 1) % 3]
        r1, r2 = points - start, points - end
        cross = np.cross(r1, r2)
        cross2 = np.einsum("ij,ij->i", cross, cross)
        l1, l2 = np.linalg.norm(r1, axis=1), np.linalg.norm(r2, axis=1)
        along = np.einsum("ij,ij->i", end - start, r1 / np.maximum(l1, 1e-300)[:, None]
                          - r2 / np.maximum(l2, 1e-300)[:, None])
        scale = np.where(cross2 > 1e-24 * np.einsum("ij,ij->i", end - start, end - start) ** 2,
                         along / np.maximum(cross2, 1e-300), 0.0)
        u -= cross * scale[:, None]
    return u / _FOUR_PI


class _Sheet:
    """Triangles (with their mirror images under the plate) as seen from many
    points: exact formulas within NEAR_FIELD panel sizes, point source and
    point doublet further away."""

    def __init__(self, triangles, ground):
        self.count = len(triangles)
        self.tri = np.concatenate([triangles, mirror(triangles)]) if ground else triangles
        self.centres = self.tri.mean(axis=1)
        area_normal = 0.5 * np.cross(self.tri[:, 1] - self.tri[:, 0], self.tri[:, 2] - self.tri[:, 0])
        self.areas = np.linalg.norm(area_normal, axis=1)
        self.normals = area_normal / self.areas[:, None]
        edges = np.stack([self.tri[:, (k + 1) % 3] - self.tri[:, k] for k in range(3)], axis=1)
        self.near = NEAR_FIELD * np.linalg.norm(edges, axis=2).max(axis=1)

    def _fold(self, values):
        """Add each mirror image's column onto its triangle's."""
        if len(self.tri) == self.count:
            return values
        return values[:, :self.count] + values[:, self.count:]

    def potentials(self, points, own=None):
        """(phi_source, phi_doublet), each (c, count)."""
        offset = points[:, None, :] - self.centres[None, :, :]
        distance = np.maximum(np.linalg.norm(offset, axis=2), 1e-12)
        phi_s = -self.areas / (_FOUR_PI * distance)
        phi_d = self.areas * np.einsum("cjk,jk->cj", offset, self.normals) / (_FOUR_PI * distance ** 3)
        rows, cols = np.nonzero(distance < self.near[None, :])
        if len(rows):
            s, d = potentials(points[rows], self.tri[cols], None if own is None else (cols == own[rows]))
            phi_s[rows, cols], phi_d[rows, cols] = s, d
        return self._fold(phi_s), self._fold(phi_d)

    def velocities(self, points, sources=True, doublets=True):
        """(source velocity, doublet velocity), each (c, count, 3) or None."""
        offset = points[:, None, :] - self.centres[None, :, :]
        distance = np.maximum(np.linalg.norm(offset, axis=2), 1e-12)[..., None]
        rows, cols = np.nonzero(distance[..., 0] < self.near[None, :])
        u_s = u_d = None
        if sources:
            u_s = self.areas[None, :, None] * offset / (_FOUR_PI * distance ** 3)
            if len(rows):
                u_s[rows, cols] = source_velocity(points[rows], self.tri[cols])
            u_s = self._fold(u_s)
        if doublets:
            moment = self.areas[:, None] * self.normals                     # point doublet far away
            m_dot_r = np.einsum("cjk,jk->cj", offset, moment)[..., None]
            u_d = (moment[None] / distance ** 3 - 3.0 * m_dot_r * offset / distance ** 5) / _FOUR_PI
            if len(rows):
                u_d[rows, cols] = doublet_velocity(points[rows], self.tri[cols])
            u_d = self._fold(u_d)
        return u_s, u_d


# -------------------------------------------------------- trailing edges, wake

def trailing_edges(panels, partner, stream=STREAM):
    """Trailing edges as (corners (E, 2) vertex indices, upper (E,), lower (E,)):
    where each wake strip leaves, the wing-top panel there and the panel on the
    wing's other side (partner: geometry.thin_partners). The corners are in
    the order that makes the wake face the top panel's side; the wake carries
    mu_upper - mu_lower."""
    faces = panels.faces
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    shared = np.nonzero(counts == 2)[0]
    if len(shared) == 0:
        return np.zeros((0, 2), int), np.zeros(0, int), np.zeros(0, int)
    order = np.argsort(inverse.ravel(), kind="stable")
    first = np.searchsorted(inverse.ravel()[order], shared)
    fa, fb = owner[order[first]], owner[order[first + 1]]

    v, n, c = panels.vertices, panels.normals, panels.centroids
    top = (partner >= 0) & (n[:, 2] > TE_UP) & ~wheels(panels, stream)
    one = top[fa] != top[fb]                               # an edge of the wing-top region
    corners, fa, fb = unique[shared][one], fa[one], fb[one]
    inside = np.where(top[fa], fa, fb)                     # the wing-top panel at the edge
    pa, pb = v[corners[:, 0]], v[corners[:, 1]]
    direction = (pb - pa) / np.linalg.norm(pb - pa, axis=1, keepdims=True)
    middle = 0.5 * (pa + pb)
    outward = np.cross(direction, n[inside])               # across the edge, away from the region
    outward *= np.sign(np.einsum("ij,ij->i", outward, middle - c[inside]))[:, None]
    is_te = (np.abs(direction @ stream) < TE_ACROSS) & (outward @ stream > TE_BACK)
    corners, upper = corners[is_te], inside[is_te]
    across = np.where(fa == inside, fb, fa)[is_te]         # the panel on the edge's other side
    # A sharp edge: the panel across it is the wing's underside, pair with it.
    # Blunt (a flat back face) or rounded: pair with the panel straight below
    # the top one. Rounded ends on coarse meshes are the least reliable; the
    # circulation guard in build() catches what goes wrong there.
    sharp = np.einsum("ij,ij->i", n[across], n[upper]) < -0.5
    lower = np.where(sharp, across, partner[upper])
    if len(corners):
        keep = _without_overlaps(v, corners, stream)       # strips that walked to the same place
        corners, upper, lower = corners[keep], upper[keep], lower[keep]
    # Corner order: the wake's normal, (b - a) x stream, on the top panel's side.
    flip = np.einsum("ij,ij->i", np.cross(v[corners[:, 1]] - v[corners[:, 0]], stream), n[upper]) < 0
    corners = np.where(flip[:, None], corners[:, ::-1], corners)
    return corners, upper, lower


def _without_overlaps(v, corners, stream, near=1.0):
    """Indices of strips to keep so that no two wake sheets lie on top of each
    other (two strips walked round a rounded edge to the same place): among
    strips within `near` mm of each other, longest first, a strip whose
    stretch across the stream is already mostly covered is dropped. Two
    sheets in one place make the equations nearly singular."""
    across = np.cross([0.0, 0.0, 1.0], stream)
    a, b = v[corners[:, 0]], v[corners[:, 1]]
    lo, hi = np.minimum(a @ across, b @ across), np.maximum(a @ across, b @ across)
    middle = 0.5 * (a + b)
    kept = []
    for i in np.argsort(-(hi - lo)):
        clash = False
        for j in kept:
            if np.linalg.norm((middle[i] - middle[j]) - ((middle[i] - middle[j]) @ across) * across) < near:
                overlap = min(hi[i], hi[j]) - max(lo[i], lo[j])
                if overlap > 0.2 * (hi[i] - lo[i]):
                    clash = True
                    break
        if not clash:
            kept.append(i)
    return np.sort(np.array(kept, int))


def wheels(panels, stream=STREAM):
    """bool (N,): panels on parts that are round about a horizontal axle
    across the stream (wheels). They turn; they don't make lift."""
    from aero.geometry import shells

    n, c, area = panels.normals, panels.centroids, panels.areas
    axle = np.cross([0.0, 0.0, 1.0], stream)
    axle /= np.linalg.norm(axle)
    out = np.zeros(len(panels), bool)
    for part in shells(panels.faces):
        tread = part[np.abs(n[part] @ axle) < WHEEL_TREAD]
        if len(tread) < 8 or area[tread].sum() < WHEEL_AREA * area[part].sum():
            continue
        # Work in the plane across the axle: the point closest to every tread
        # normal's line (least squares) is the axle, if there is one.
        d = n[tread] - np.outer(n[tread] @ axle, axle)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        q = c[tread] - np.outer(c[tread] @ axle, axle)
        across = np.eye(3)[None] - d[:, :, None] * d[:, None, :] - axle[None, :, None] * axle[None, None, :]
        A, b = across.sum(axis=0), np.einsum("kij,kj->i", across, q)
        centre = np.linalg.lstsq(A + np.outer(axle, axle), b, rcond=None)[0]
        miss = np.linalg.norm(np.einsum("kij,kj->ki", across, centre - q), axis=1)
        radius = np.linalg.norm(q - centre + np.outer((q - centre) @ axle, axle), axis=1)
        if np.sqrt(np.average(miss ** 2, weights=area[tread])) < WHEEL_ROUND * np.average(radius, weights=area[tread]):
            out[part] = True
    # A small part inside a wheel (its hub) belongs to the wheel.
    if out.any():
        boxes = [(c[part].min(axis=0), c[part].max(axis=0)) for part in shells(panels.faces) if out[part].all()]
        for part in shells(panels.faces):
            lo, hi = c[part].min(axis=0), c[part].max(axis=0)
            if not out[part].all() and any(np.all(lo >= blo) and np.all(hi <= bhi) for blo, bhi in boxes):
                out[part] = True
    return out


def wake_lengths(panels, corners, length, stream=STREAM):
    """How far each wake strip can trail before it would run into the model
    (another wing, a wheel, the bodywork): it stops WAKE_CLEARANCE short of
    the first surface hit straight downstream of its trailing edge, or runs
    the full `length`. (E,)."""
    start = panels.vertices[corners].mean(axis=1) + WAKE_START * stream
    tri = panels.triangles
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(stream, e2)                                                # Moller-Trumbore, one direction
    det = e1 @ p if p.ndim == 1 else np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = start[:, None, :] - tri[None, :, 0]
    u = np.einsum("ejk,jk->ej", s, p) * inv
    q = np.cross(s, e1[None])
    v = np.einsum("k,ejk->ej", stream, q) * inv
    t = np.einsum("ejk,jk->ej", q, e2) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    first = np.where(hit, t, np.inf).min(axis=1)
    return np.minimum(length, first - WAKE_CLEARANCE)


def wake_triangles(panels, corners, lengths, stream=STREAM):
    """Two triangles per trailing edge, trailing lengths[e] downstream, facing
    the 'upper' side (so their doublet is mu_upper - mu_lower). (2E, 3, 3)."""
    a, b = panels.vertices[corners[:, 0]], panels.vertices[corners[:, 1]]
    far = np.asarray(lengths, np.float64)[:, None] * stream
    first = np.stack([a, b, b + far], axis=1)
    second = np.stack([a, b + far, a + far], axis=1)
    return np.concatenate([first, second])


# ------------------------------------------------------------ surface gradient

def gradient_operator(panels, rings):
    """Least-squares gradient of a per-panel value along the surface:
    (neighbours (N, K), weights (N, K, 3), fill (N, K)) with
        grad_i = sum_k weights[i, k] * (value[neighbours[i, k]] - value[i]),
    then, for panels whose neighbours are too poorly spread to fit a slope
    (slivers, fan corners: the decimated print meshes have plenty),
        grad_i = sum_k fill[i, k] * grad[neighbours[i, k]]
    (the area-weighted mean of their well-fitted neighbours'). Only neighbours
    on the same smooth surface count (not across a trailing edge or a sharp
    corner). Everything stays linear in the value, so the per-panel 3x3
    matrices survive."""
    c, n = panels.centroids, panels.normals
    nbr = rings
    d = c[nbr] - c[:, None, :]
    d -= np.einsum("ikj,ij->ik", d, n)[..., None] * n[:, None, :]            # into the tangent plane
    same = (np.einsum("ikj,ij->ik", n[nbr], n) > SAME_SURFACE) & (nbr != np.arange(len(c))[:, None])
    # a tangent basis per panel
    helper = np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    t1 = np.cross(n, helper)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(n, t1)
    x, y = np.einsum("ikj,ij->ik", d, t1), np.einsum("ikj,ij->ik", d, t2)
    w = np.where(same, 1.0 / np.maximum(x * x + y * y, 1e-300), 0.0)        # nearer neighbours count more
    sxx, sxy, syy = (w * x * x).sum(1), (w * x * y).sum(1), (w * y * y).sum(1)
    det = sxx * syy - sxy * sxy
    ok = (same.sum(1) >= 2) & (det > GRADIENT_SPREAD * (sxx + syy) ** 2)
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    gx = w * (syy[:, None] * x - sxy[:, None] * y) * inv[:, None]          # d(value)/d(t1) weights
    gy = w * (sxx[:, None] * y - sxy[:, None] * x) * inv[:, None]
    weights = gx[..., None] * t1[:, None, :] + gy[..., None] * t2[:, None, :]
    fill = np.where(~ok[:, None] & same & ok[nbr], panels.areas[nbr], 0.0)
    fill /= np.maximum(fill.sum(axis=1, keepdims=True), 1e-300)
    return nbr, weights, fill


def surface_gradient(nbr, weights, fill, values):
    """grad of per-panel values (N, m) -> (N, 3, m)."""
    diff = values[nbr] - values[:, None, :]
    grad = np.einsum("ika,ikm->iam", weights, diff)
    filled = fill.any(axis=1)
    grad[filled] = np.einsum("ik,ikam->iam", fill[filled], grad[nbr[filled]])
    return grad


# ----------------------------------------------------------------------- build

class LiftingParts:
    """What the lifting build keeps besides the per-panel matrices: the doublet
    strengths per unit wind (plain: no wake; lift: with it), and the wake."""

    def __init__(self, mu, mu_lift, wake, upper, lower):
        self.mu = np.asarray(mu, np.float64)              # (N, 3)
        self.mu_lift = np.asarray(mu_lift, np.float64)    # (N, 3)
        self.wake = np.asarray(wake, np.float64)          # (2E, 3, 3)
        self.upper = np.asarray(upper, int)               # (E,)
        self.lower = np.asarray(lower, int)


def build(panels, corners, upper, lower, rings, ground=True, tick=None):
    """The plain and lifting solutions. Returns (M_plain, M_lift, LiftingParts)
    with M (N, 3, 3): surface velocity = M_i @ wind."""
    n = len(panels)
    normals = panels.normals
    size = float(np.ptp(panels.vertices, axis=0).max())
    body = _Sheet(panels.triangles, ground)
    lengths = wake_lengths(panels, corners, WAKE_LENGTH * size)
    keep = lengths > WAKE_CLEARANCE                    # a strip blocked right at its edge sheds nothing
    corners, upper, lower, lengths = corners[keep], upper[keep], lower[keep], lengths[keep]
    wake = wake_triangles(panels, corners, lengths)
    wake_sheet = _Sheet(wake, ground)
    edges = len(corners)
    chunks = [(s, min(s + ROWS_PER_CHUNK, n)) for s in range(0, n, ROWS_PER_CHUNK)]

    D = np.empty((n, n))                     # doublet j -> potential just inside panel i
    R = np.empty((n, 3))                     # the sources' part, per unit wind (sigma = -n . wind)
    W = np.zeros((n, edges))                 # wake strip e -> potential inside panel i
    for step, (start, stop) in enumerate(chunks):
        points = panels.centroids[start:stop]
        phi_s, phi_d = body.potentials(points, own=np.arange(start, stop))
        D[start:stop] = phi_d
        R[start:stop] = phi_s @ -normals
        if edges:
            _, phi_w = wake_sheet.potentials(points)
            W[start:stop] = phi_w[:, :edges] + phi_w[:, edges:]          # two triangles per strip
        if tick:
            tick((step + 1) / len(chunks))

    mu = np.linalg.solve(D, -R)              # still air inside: D mu + R wind = 0
    mu_lift = mu
    keep = np.ones(edges, bool)
    while keep.any():
        D_lift = D.copy()
        np.add.at(D_lift.T, upper[keep], W[:, keep].T)          # Kutta: mu_wake = mu_upper - mu_lower
        np.add.at(D_lift.T, lower[keep], -W[:, keep].T)
        mu_lift = np.linalg.solve(D_lift, -R)
        del D_lift
        gamma = np.abs((mu_lift[upper] - mu_lift[lower]) @ STREAM)
        too_strong = keep & (gamma > GAMMA_LIMIT * size)
        if not too_strong.any():
            break
        keep &= ~too_strong
    if not keep.any():
        mu_lift = mu
    corners, upper, lower = corners[keep], upper[keep], lower[keep]
    wake = wake_triangles(panels, corners, lengths[keep]) if len(corners) else np.zeros((0, 3, 3))

    # Surface velocity: the wind along the surface plus the gradient of mu.
    operator = gradient_operator(panels, rings)
    along = np.eye(3)[None] - normals[:, :, None] * normals[:, None, :]
    M = along + surface_gradient(*operator, mu)
    M_lift = along + surface_gradient(*operator, mu_lift)
    return M, M_lift, LiftingParts(mu, mu_lift, wake, upper, lower)


def velocity_matrices(panels, parts, points, ground=True, chunk=64, progress=None, cancel=None):
    """(G_plain, G_lift), each (c, 3, 3): velocity = G @ wind at points off the
    surface, from the sources, the doublets and (lifting) the wake."""
    from aero.panel import Cancelled

    body = _Sheet(panels.triangles, ground)
    wake = _Sheet(parts.wake, ground)
    edges = len(parts.upper)
    sigma = -panels.normals                                  # sigma = -n . wind, per unit wind
    mu_wake = parts.mu_lift[parts.upper] - parts.mu_lift[parts.lower]
    points = np.asarray(points, np.float64)
    G = np.empty((len(points), 3, 3))
    G_lift = np.empty((len(points), 3, 3))
    for start in range(0, len(points), chunk):
        if cancel is not None and cancel():
            raise Cancelled()
        stop = min(start + chunk, len(points))
        p = points[start:stop]
        u_s, u_d = body.velocities(p)
        base = np.eye(3) + np.einsum("cjk,jl->ckl", u_s, sigma)
        G[start:stop] = base + np.einsum("cjk,jl->ckl", u_d, parts.mu)
        G_lift[start:stop] = base + np.einsum("cjk,jl->ckl", u_d, parts.mu_lift)
        if edges:
            _, u_w = wake.velocities(p, sources=False)
            u_w = u_w[:, :edges] + u_w[:, edges:]
            G_lift[start:stop] += np.einsum("cjk,jl->ckl", u_w, mu_wake)
        if progress is not None:
            progress(stop / len(points))
    return G, G_lift
