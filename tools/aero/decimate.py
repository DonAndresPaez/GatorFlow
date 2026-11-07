'''decimate.py: turn a detailed STL into one the panel solver can use.

Run (from tools/):
    python -m aero.decimate SOURCE.stl OUT.stl                 to about 6000 triangles
    python -m aero.decimate SOURCE.stl OUT.stl --triangles 4000

Needs: pip install fast-simplification  (only here; TouchDesigner never runs this)

The panel method builds an N x N matrix, so it wants a few thousand triangles;
printable models often have a hundred thousand. This:
  1. splits the mesh into its separate shells (a body, separate wheels, ...);
  2. drops shells facing inwards: cavities inside another shell, whose
     surfaces are not in the air;
  3. simplifies each shell on its own (quadric edge collapse), sharing the
     triangle budget by surface area, so every shell stays a closed solid;
  4. checks every result is still closed and facing out, and how much its
     volume changed, and refuses to write a broken mesh.

Thin parts (wings, fins) keep their shape only as far as the budget allows:
look at the report, and raise --triangles if a thin shell's volume changed a lot.
'''

import argparse

import numpy as np

from aero.geometry import is_closed, merge_corners, read_stl, shells, signed_volume, write_stl

MIN_PER_SHELL = 300             # even a small shell keeps this many triangles


def decimate(triangles, target=6000, log=print):
    """Simplified triangles (T, 3, 3), every shell closed and facing out."""
    import fast_simplification

    vertices, faces = merge_corners(triangles)
    pieces = []
    for index in shells(faces):
        tri = vertices[faces[index]]
        volume = signed_volume(tri)
        if volume <= 0:
            log(f"  dropped a shell of {len(index)} triangles facing inwards (a cavity, volume {volume:.1f} mm^3)")
            continue
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        pieces.append((index, volume, area))

    total_area = sum(area for _, _, area in pieces)
    out = []
    for index, volume, area in pieces:
        used, local = np.unique(faces[index], return_inverse=True)
        points, tris = vertices[used], local.reshape(-1, 3)
        want = max(MIN_PER_SHELL, int(round(target * area / total_area)))
        if want < len(tris):
            points, tris = fast_simplification.simplify(points, tris.astype(np.int32), target_count=want)
        result = points[tris]
        new_volume = signed_volume(result)
        closed = is_closed(np.asarray(tris))
        log(f"  shell {len(index):6d} -> {len(tris):5d} triangles, volume {volume:9.1f} -> {new_volume:9.1f} mm^3 "
            f"({100 * (new_volume / volume - 1):+.1f}%), {'closed' if closed else 'NOT CLOSED'}")
        if not closed or new_volume <= 0:
            raise ValueError("a simplified shell is open or inside out: try more --triangles")
        out.append(result)
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--triangles", type=int, default=6000, help="total to aim for (default 6000)")
    args = ap.parse_args()

    triangles = read_stl(args.source)
    print(f"{args.source}: {len(triangles)} triangles")
    result = decimate(triangles, args.triangles)
    write_stl(args.out, result, header=f"decimated by tools/aero/decimate.py to {len(result)} triangles")
    print(f"wrote {args.out}: {len(result)} triangles")


if __name__ == "__main__":
    main()
