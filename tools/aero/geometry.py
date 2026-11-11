'''geometry.py: an STL turned into the panels the solvers work on.

Imported by: aero/panel.py, aero/pipeline.py, aero/train.py, tests
Run directly: no

The plate is the ground. Panels lying flat on it (the model's base) are not in
the air, so they are dropped, and the solvers mirror what is left below the
plate. Body + mirror image is a closed surface even though the base is gone.

Two frames:
  STL frame    the coordinates in the file (what TouchDesigner's File In SOP shows)
  plate frame  after mountOffsetMm / mountRotationDeg: X right, Y far edge, Z up,
               plate top at Z = 0. The solvers work here, the ground is Z = 0.
'''

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Panels are treated as the base (on the plate, not wetted) when their centre is
# this close to the plate and they face down.
BASE_TOLERANCE_MM = 0.5
BASE_NORMAL_Z = -0.9

# Triangles smaller than this fraction of the mean are slivers left by the CAD
# export; they add nothing and make the influence matrix badly conditioned.
MIN_AREA_FRACTION = 1e-6

# Coarser STLs are subdivided up to at least this many triangles (refine()).
MIN_PANELS = 1500

# Separate shells closer than this (a print-in-place wheel and the car it turns
# on) leave a slit no real air flows through, but potential flow would rush
# through it (Cp -145 on the F1 car's 0.46 mm wheel gaps). Faces across such a
# gap are not wetted.
GAP_MM = 1.0
MIN_PART_PANELS = 4             # smaller pieces left over after removing gap faces are dropped

# A wing (or fin, or floor) for the wing-loading display: a part whose two
# sides face away from each other closer together than this, or than this
# fraction of the model size, whichever is bigger.
THIN_MM, THIN_FRACTION = 4.0, 0.04

_STL_RECORD = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


def read_stl(path):
    """Triangles of a binary or ASCII STL, shape (T, 3, 3), float64, file units."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = int(np.frombuffer(data, "<u4", 1, 80)[0])
        if len(data) == 84 + 50 * count:
            return np.frombuffer(data, _STL_RECORD, count, 84)["v"].astype(np.float64)
    # ASCII: every "vertex x y z" line, three per facet.
    numbers = [line.split()[1:4] for line in data.decode("ascii", "replace").splitlines()
               if line.strip().startswith("vertex")]
    if not numbers or len(numbers) % 3:
        raise ValueError(f"{path}: not a readable STL")
    return np.array(numbers, np.float64).reshape(-1, 3, 3)


def mount_transform(offset_mm=(0, 0, 0), rotation_deg=0.0, scale=1.0):
    """4x4 STL frame -> plate frame, as load_setup.py's `mount` Transform SOP does
    (scale, rotate about Z, then move). The offset is for the STL at scale 1 and
    grows with it: the placed model is scaled about the plate centre, so a model
    centred on the plate with its wheels on it stays that way."""
    a = np.radians(rotation_deg)
    T = np.eye(4)
    T[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    T[:3, :3] *= scale
    T[:3, 3] = np.asarray(offset_mm, float) * scale
    return T


@dataclass
class Panels:
    """The wetted surface, as triangles in the plate frame."""
    vertices: np.ndarray        # (V, 3) plate frame, mm, shared corners merged
    faces: np.ndarray           # (N, 3) vertex indices, counter-clockwise seen from outside
    vertices_stl: np.ndarray    # (V, 3) the same corners in the STL frame (for TouchDesigner)
    key: str                    # changes whenever the geometry the solver sees changes

    @property
    def triangles(self):
        return self.vertices[self.faces]

    @property
    def centroids(self):
        return self.triangles.mean(axis=1)

    @property
    def area_normals(self):
        t = self.triangles
        return 0.5 * np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])

    @property
    def areas(self):
        return np.linalg.norm(self.area_normals, axis=1)

    @property
    def normals(self):
        return self.area_normals / self.areas[:, None]

    def __len__(self):
        return len(self.faces)

    def to_vertices(self, per_panel, use=None):
        """Per-panel values -> per-vertex values, area-weighted over the panels
        that share each corner (smooth colours instead of one colour per facet).
        use: bool (N,), only these panels count; corners touching none get 0."""
        weights = np.repeat(self.areas if use is None else self.areas * use, 3)
        index = self.faces.ravel()
        values = np.repeat(np.asarray(per_panel, np.float64), 3)
        total = np.bincount(index, weights * values, len(self.vertices))
        norm = np.bincount(index, weights, len(self.vertices))
        # Corners only touched by base panels have no wetted neighbour.
        return np.divide(total, norm, out=np.zeros_like(total), where=norm > 0)


def refine(triangles, min_count=MIN_PANELS):
    """Split every triangle into four at its edge midpoints until there are at
    least min_count. A coarse STL (a cube is 12 triangles) gives the panel
    method too few panels and TouchDesigner too few points to colour. Flat
    faces stay exactly flat, winding is kept, and neighbours still share
    corners. Meshes already that fine are returned unchanged."""
    tri = np.asarray(triangles, np.float64)
    while len(tri) < min_count:
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate([np.stack(t, axis=1) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))])
    return tri


def merge_corners(triangles):
    """(T, 3, 3) triangles -> (V, 3) vertices and (T, 3) indices. Corners within
    a millionth of the model size are the same corner."""
    flat = np.asarray(triangles, np.float64).reshape(-1, 3)
    scale = float(np.ptp(flat, axis=0).max())
    _, first, inverse = np.unique(np.round(flat / (scale * 1e-6)).astype(np.int64), axis=0,
                                  return_index=True, return_inverse=True)
    return flat[first], inverse.reshape(-1, 3)


def write_stl(path, triangles, header="written by tools/aero/geometry.py"):
    """Binary STL of (T, 3, 3) triangles (facet normals from the winding)."""
    tri = np.asarray(triangles, np.float64)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)
    record = np.zeros(len(tri), _STL_RECORD)
    record["normal"], record["v"] = n, tri
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(header.encode()[:80].ljust(80, b" ") + np.uint32(len(tri)).tobytes() + record.tobytes())


def shells(faces):
    """Split a mesh into its separate pieces: a list of face-index arrays,
    one per set of faces connected through shared corners."""
    faces = np.asarray(faces)
    label = np.arange(faces.max() + 1)
    while True:                     # spread the smallest label across every face until nothing changes
        smallest = label[faces].min(axis=1)
        new = label.copy()
        np.minimum.at(new, faces.ravel(), np.repeat(smallest, 3))
        new = new[new]              # follow labels pointing at relabelled corners
        if np.array_equal(new, label):
            break
        label = new
    per_face = label[faces[:, 0]]
    return [np.nonzero(per_face == k)[0] for k in np.unique(per_face)]


def is_closed(faces):
    """True when the mesh has no holes: every edge is shared by an even number
    of triangles (two; four where two surfaces pinch together along an edge)."""
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts % 2 == 0))


def signed_volume(triangles):
    """Enclosed volume (divergence theorem): negative when the triangles face inwards."""
    t = np.asarray(triangles, np.float64)
    return float(np.einsum("ij,ij->", t[:, 0], np.cross(t[:, 1], t[:, 2])) / 6.0)


def write_obj(stl_path, obj_path, transform=None):
    """The STL as an OBJ with shared corners, refined like the panels so every
    panel corner is a point to colour. Same units and frame as the STL, or
    moved by a 4x4 `transform`. TouchDesigner's File In SOP reads OBJ but not STL."""
    vertices, faces = merge_corners(refine(read_stl(stl_path)))
    if transform is not None:
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
    lines = [f"# {Path(stl_path).name}, written by tools/aero/geometry.py (refined like the solver's panels)\n"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces]
    Path(obj_path).parent.mkdir(parents=True, exist_ok=True)
    Path(obj_path).write_text("".join(lines))


def obj_for(stl_path, folder):
    """folder/<stl name>.obj, rewritten whenever the STL is newer."""
    stl_path = Path(stl_path)
    obj_path = Path(folder) / (stl_path.stem + ".obj")
    if not obj_path.exists() or obj_path.stat().st_mtime < stl_path.stat().st_mtime:
        write_obj(stl_path, obj_path)
    return obj_path


def _in_gaps(vertices, faces, gap=GAP_MM, chunk=512):
    """Faces that look straight at a face of ANOTHER shell less than `gap`
    away. Only between shells: a crease in one shell is not a gap, and the
    two sides of a thin wing face away from each other."""
    pieces = shells(faces)
    out = np.zeros(len(faces), bool)
    if len(pieces) < 2:
        return out
    t = vertices[faces]
    c = t.mean(axis=1)
    area_normal = 0.5 * np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    size = np.sqrt(2 * np.linalg.norm(area_normal, axis=1))
    n = area_normal / np.maximum(np.linalg.norm(area_normal, axis=1, keepdims=True), 1e-300)
    def near(index, other):
        # A face can only face the other shell across a gap if its centre is
        # within gap + (the larger of the two faces) of that shell's box.
        reach = gap + np.maximum(size[index], size[other].max())[:, None]
        lo, hi = c[other].min(axis=0), c[other].max(axis=0)
        return index[np.all((c[index] >= lo - reach) & (c[index] <= hi + reach), axis=1)]

    for a, shell_a in enumerate(pieces):
        for shell_b in pieces[:a] + pieces[a + 1:]:
            mine, theirs = near(shell_a, shell_b), near(shell_b, shell_a)
            for s in range(0, len(mine), chunk):
                i = mine[s:s + chunk]
                d = c[None, theirs, :] - c[i, None, :]                     # this face -> the other shell's
                ahead = np.einsum("ijk,ik->ij", d, n[i])                   # distance in front of this face
                aside = np.linalg.norm(d - ahead[..., None] * n[i, None, :], axis=2)
                out[i] |= ((ahead > 0) & (ahead < gap)
                           & (aside < np.maximum(size[i, None], size[None, theirs]))
                           & ((n[i] @ n[theirs].T) < -0.7)).any(axis=1)  # facing each other
    return out


def panel_rings(panels, limit):
    """Each panel's neighbourhood: the panels sharing a corner with it, nearest
    first, at most `limit` (fan corners can have dozens). (N, K), padded with
    the panel itself."""
    faces, centroids = panels.faces, panels.centroids
    by_vertex = [[] for _ in range(len(panels.vertices))]
    for t, f in enumerate(faces):
        for v in f:
            by_vertex[v].append(t)
    rings = []
    for t, f in enumerate(faces):
        ring = np.array(sorted(set(by_vertex[f[0]] + by_vertex[f[1]] + by_vertex[f[2]])))
        ring = ring[np.argsort(np.linalg.norm(centroids[ring] - centroids[t], axis=1))][:limit]
        rings.append(list(ring))
    width = max(len(r) for r in rings)
    return np.array([r + [t] * (width - len(r)) for t, r in enumerate(rings)])


def thin_partners(panels, thickness=None, chunk=512):
    """For each panel on a thin part, the panel on its other side: same shell,
    facing the opposite way, straight behind it within `thickness` mm
    (default: THIN_MM or THIN_FRACTION of the model, whichever is bigger).
    (N,) indices, -1 where the panel is not on a thin part."""
    c, n = panels.centroids, panels.normals
    size = np.sqrt(2 * panels.areas)
    if thickness is None:
        thickness = max(THIN_MM, THIN_FRACTION * float(np.ptp(panels.vertices, axis=0).max()))
    shell = np.empty(len(panels), int)
    for k, index in enumerate(shells(panels.faces)):
        shell[index] = k
    partner = np.full(len(panels), -1)
    for s in range(0, len(panels), chunk):
        i = np.arange(s, min(s + chunk, len(panels)))
        d = c[None, :, :] - c[i, None, :]
        behind = -np.einsum("ijk,ik->ij", d, n[i])                   # distance straight behind the panel
        aside = np.linalg.norm(d + behind[..., None] * n[i, None, :], axis=2)
        ok = ((behind > 0) & (behind < thickness) & (aside < np.maximum(size[i, None], size[None, :]))
              & ((n[i] @ n.T) < -0.7) & (shell[i, None] == shell[None, :]))
        score = np.where(ok, behind + aside, np.inf)
        best = np.argmin(score, axis=1)
        partner[i] = np.where(np.isfinite(score[np.arange(len(i)), best]), best, -1)
    return partner


def panels_from_triangles(triangles_stl, mount=np.eye(4), ground=True, label=""):
    """Merge corners, drop slivers and (with ground) the base, check the normals point out."""
    vertices_stl, faces = merge_corners(triangles_stl)
    vertices = vertices_stl @ mount[:3, :3].T + mount[:3, 3]

    t = vertices[faces]
    area_normal = 0.5 * np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    area = np.linalg.norm(area_normal, axis=1)
    keep = area > MIN_AREA_FRACTION * area.mean()
    # Corners merged into one (distinct indices) are degenerate too.
    keep &= (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])

    # The divergence theorem gives the enclosed volume, which is negative when
    # the file's triangles wind inward; then the faces are flipped. Measured before
    # the base is removed, while the surface is still closed.
    volume = np.sum(t[keep].mean(axis=1) * area_normal[keep]) / 3.0
    if volume < 0:
        faces = faces[:, [0, 2, 1]]
        area_normal = -area_normal

    if ground:
        centre_z = t[:, :, 2].mean(axis=1)
        normal_z = area_normal[:, 2] / np.maximum(area, 1e-300)
        keep &= ~((centre_z < BASE_TOLERANCE_MM) & (normal_z < BASE_NORMAL_Z))
        if vertices[:, 2].min() < -BASE_TOLERANCE_MM:
            print(f"aero: {label or 'model'} reaches {vertices[:, 2].min():.1f} mm below the plate; "
                  "check mountOffsetMm (the ground mirror assumes nothing is under the plate)")

    faces = faces[keep]
    gaps = _in_gaps(vertices, faces)
    if gaps.any():
        # Taking out gap faces can leave a few loose triangles behind (round a
        # wheel's hub): flaps hanging in the air, which the solver doesn't want.
        faces = faces[~gaps]
        keep = np.ones(len(faces), bool)
        for part in shells(faces):
            if len(part) < MIN_PART_PANELS:
                keep[part] = False
        faces = faces[keep]
    digest = hashlib.sha1()
    for array in (vertices, faces, np.array([ground], np.int8)):
        digest.update(np.ascontiguousarray(array).tobytes())
    return Panels(vertices, faces, vertices_stl, digest.hexdigest()[:16])


def load_panels(stl_path, mount=np.eye(4), ground=True, units="mm"):
    """Panels of an STL (refined if coarse), in mm in the plate frame."""
    triangles = refine(read_stl(stl_path))
    if units == "m":
        triangles = triangles * 1000.0
    return panels_from_triangles(triangles, mount, ground, label=Path(stl_path).name)
