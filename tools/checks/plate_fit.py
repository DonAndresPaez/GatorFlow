'''plate_fit.py: which marker is off, and where the markers really are.

Run: python -m checks.plate_fit              (from tools/; plate on the table, all markers in view)
     python -m checks.plate_fit --write      (also puts the measured positions into setup/plate.yml)
Keys: SPACE = start / restart collecting, q = quit

The fit error in live_view is one number for the whole plate. This splits it
per marker, so you can see whether ONE marker is off (it was measured or stuck
wrong) or ALL of them are (size, lens, or a bent plate).

Then it measures the layout with the camera instead of a ruler: for each
frame it fits the plate pose, projects every detected corner back onto the
plate's surface, and reads off where each marker's centre, turn and size
really are. It repeats that with the improved numbers until they stop
changing, and keeps your plate frame (its centre and axes stay where plate.yml
put them; only the markers' positions RELATIVE to each other are corrected).

The ruler is your plate.yml: one camera can't tell a plate 1% bigger and 1%
farther away from this one, so the measured layout is scaled to have the same
overall spread as the centres you measured by hand (over ~400 mm, a mm of tape
error is 0.25%). The marker size then comes from the camera too, so the
layout agrees with exactly what the detector sees.

What it can't fix: a plate or a marker that isn't flat, and a lens calibration
that is off. If the error stays high after --write, look at those.

Rerun calibration.set_origin after writing new positions.
'''

import argparse
import re

import cv2
import numpy as np

from common import paths
from common.camera import load_intrinsics, open_camera
from common.plate import LAYOUT, MarkerLayout, MarkerTracker
from tracking.tracker import LatestFrame, detect_markers

FRAMES = 60
MAX_ITERATIONS = 40
CONVERGED_MM = 0.01
VIEW_WIDTH = 1280


# ---------------------------------------------------------------- the math (no camera; tested)

def solve_pose(layout, detections, K, dist):
    """Plate pose from {id: 4x2 corners} with this layout, no rejection."""
    obj = np.vstack([layout.corners(layout.ids.index(i)) for i in detections]).astype(np.float64)
    img = np.vstack([detections[i] for i in detections]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    return rvec, tvec


def residuals_px(layout, detections, K, dist, rvec, tvec):
    """RMS distance (px) between seen and predicted corners, per marker id."""
    out = {}
    for i, seen in detections.items():
        predicted, _ = cv2.projectPoints(layout.corners(layout.ids.index(i)).astype(np.float64), rvec, tvec, K, dist)
        out[i] = float(np.sqrt(np.mean(np.sum((predicted.reshape(4, 2) - seen) ** 2, axis=1))))
    return out


def corners_on_plate(corners_px, K, dist, rvec, tvec):
    """Where the camera rays through these pixels hit the plate surface (z = 0), in plate mm."""
    rays = cv2.undistortPoints(corners_px.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)
    R, _ = cv2.Rodrigues(rvec)
    origin = -R.T @ tvec.ravel()                          # camera centre in plate coordinates
    directions = (R.T @ np.column_stack([rays, np.ones(len(rays))]).T).T
    s = -origin[2] / directions[:, 2]
    return (origin + s[:, None] * directions)[:, :2]


def marker_geometry(c):
    """Centre, turn (deg) and side (mm) of one marker from its 4 corners in
    ArUco order: top-left, top-right, bottom-right, bottom-left."""
    centre = c.mean(axis=0)
    across = (c[1] - c[0]) + (c[2] - c[3])               # left -> right, along +x when not turned
    up = (c[0] - c[3]) + (c[1] - c[2])                   # bottom -> top, along +y
    angle = np.degrees(np.arctan2(across[1] - up[0], across[0] + up[1]))
    side = np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])
    return centre, angle, side


def keep_plate_frame(new_centres, new_angles, old_centres):
    """Rigidly move the measured layout so it best matches the old one: the
    plate's centre and axes stay what plate.yml defined; only the markers'
    positions relative to each other change. (2D Kabsch, no scaling.)"""
    a, b = new_centres - new_centres.mean(0), old_centres - old_centres.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    R = (u @ vt).T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = (u @ vt).T
    turn = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    centres = (R @ a.T).T + old_centres.mean(0)
    return centres, (new_angles + turn + 180) % 360 - 180


def spread(centres):
    """RMS distance of the centres from their middle: the plate's overall size."""
    return float(np.sqrt(np.mean(np.sum((centres - centres.mean(0)) ** 2, axis=1))))


def fit_layout(frames, layout, K, dist, max_iterations=MAX_ITERATIONS):
    """Measured layout from frames of {id: 4x2 corners} (all markers seen).
    Returns the layout and each marker's size as seen."""
    current = layout
    for _ in range(max_iterations):
        per_marker = {i: [] for i in layout.ids}
        for detections in frames:
            rvec, tvec = solve_pose(current, detections, K, dist)
            for i, c in detections.items():
                per_marker[i].append(marker_geometry(corners_on_plate(c, K, dist, rvec, tvec)))
        centres = np.array([np.median([g[0] for g in per_marker[i]], axis=0) for i in layout.ids])
        angles = np.array([np.median([g[1] for g in per_marker[i]]) for i in layout.ids])
        sides = np.array([np.median([g[2] for g in per_marker[i]]) for i in layout.ids])
        # Ruler: keep the overall spread of the hand-measured centres.
        scale = spread(layout.centers_mm) / spread(centres)
        centres = (centres - centres.mean(0)) * scale + centres.mean(0)
        sides = sides * scale
        centres, angles = keep_plate_frame(centres, angles, layout.centers_mm)
        change = np.abs(centres - current.centers_mm).max()
        current = MarkerLayout(layout.dictionary_name, float(np.mean(sides)), list(layout.ids), centres, angles)
        if change < CONVERGED_MM:
            break
    return current, sides


def mean_residuals(frames, layout, K, dist):
    total = {i: [] for i in layout.ids}
    for detections in frames:
        rvec, tvec = solve_pose(layout, detections, K, dist)
        for i, r in residuals_px(layout, detections, K, dist, rvec, tvec).items():
            total[i].append(r)
    return {i: float(np.mean(v)) for i, v in total.items()}


# ---------------------------------------------------------------- writing plate.yml

def write_positions(layout):
    """Replace only markerSizeMm and the marker lines in setup/plate.yml; comments stay."""
    text = paths.PLATE_FILE.read_text()
    text, count = re.subn(r"^markerSizeMm:.*$", f"markerSizeMm: {layout.size_mm:.2f}", text, flags=re.M)
    if count != 1:
        raise RuntimeError(f"could not find markerSizeMm in {paths.PLATE_FILE}")
    for index, marker_id in enumerate(layout.ids):
        x, y = layout.centers_mm[index]
        line = (f"   - {{ id: {marker_id}, xMm: {x:7.1f}, yMm: {y:7.1f}, "
                f"rotationDeg: {layout.rotations_deg[index]:5.2f} }}")
        text, count = re.subn(rf"^\s*-\s*\{{\s*id:\s*{marker_id}\s*,.*\}}.*$", line, text, flags=re.M)
        if count != 1:
            raise RuntimeError(f"could not find the line for id {marker_id} in {paths.PLATE_FILE}")
    paths.PLATE_FILE.write_text(text)


# ---------------------------------------------------------------- live

def draw(frame, detections, res, mm_per_px, status):
    scale = min(1.0, VIEW_WIDTH / frame.shape[1])
    view = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for i, c in detections.items():
        pts = (c * scale).astype(np.int32)
        r = res.get(i)
        color = (0, 255, 0) if r is None or r * mm_per_px < 1.0 else (0, 200, 255) if r * mm_per_px < 2.0 else (0, 0, 255)
        cv2.polylines(view, [pts], True, color, 2)
        label = f"id {i}" + (f"  {r:.1f} px / {r * mm_per_px:.1f} mm" if r is not None else "")
        x, y = pts[:, 0].min(), pts[:, 1].min() - 8
        cv2.putText(view, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    for row, text in enumerate(status):
        cv2.putText(view, text, (12, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (12, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="put the measured positions into setup/plate.yml")
    ap.add_argument("--frames", type=int, default=FRAMES)
    args = ap.parse_args()

    lens = load_intrinsics()
    tracker = MarkerTracker(lens.K, lens.dist)
    camera = open_camera()
    reader = LatestFrame(camera)
    window = "plate fit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, VIEW_WIDTH, VIEW_WIDTH * 9 // 16)
    print(f"Plate on the table, all {len(LAYOUT.ids)} markers in view. SPACE starts collecting {args.frames} frames, q quits.")
    print("Moving or tilting the plate a little while collecting is fine, even better: it averages out the lens.")

    frames, collecting, seen = [], False, 0
    try:
        while len(frames) < args.frames:
            frame, seen = reader.newest(seen)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                frames, collecting = [], True
            if frame is None:
                continue
            corners, ids = detect_markers(tracker, frame)
            detections = {int(i): c.reshape(4, 2).astype(np.float64)
                          for c, i in zip(corners, np.ravel(ids) if ids is not None else []) if int(i) in LAYOUT.ids}
            res, mm_per_px = {}, 0.0
            if detections:
                rvec, tvec = solve_pose(LAYOUT, detections, lens.K, lens.dist)
                res = residuals_px(LAYOUT, detections, lens.K, lens.dist, rvec, tvec)
                mm_per_px = float(np.linalg.norm(tvec)) / lens.K[0, 0]
            complete = len(detections) == len(LAYOUT.ids)
            if collecting and complete:
                frames.append(detections)
            overall = np.sqrt(np.mean(np.square(list(res.values())))) if res else 0
            status = [f"fit error {overall:.1f} px ({overall * mm_per_px:.1f} mm)   green < 1 mm, orange < 2 mm, red worse",
                      (f"collecting {len(frames)}/{args.frames}" if collecting else "SPACE = collect")
                      + ("" if complete else f"   sees {sorted(detections)}, needs all {LAYOUT.ids}")]
            cv2.imshow(window, draw(frame, detections, res, mm_per_px, status))
    finally:
        reader.stop()
        camera.release()
        cv2.destroyAllWindows()

    if len(frames) < 10:
        print("Not enough frames with every marker in view. Nothing measured.")
        return

    before = mean_residuals(frames, LAYOUT, lens.K, lens.dist)
    fitted, sides = fit_layout(frames, LAYOUT, lens.K, lens.dist)
    after = mean_residuals(frames, fitted, lens.K, lens.dist)

    print(f"\n{'id':>3} {'error now':>10} {'after fit':>10}   {'plate.yml (x, y, turn)':>26}   {'measured (x, y, turn)':>26}   size seen")
    for k, i in enumerate(LAYOUT.ids):
        ox, oy = LAYOUT.centers_mm[k]
        nx, ny = fitted.centers_mm[k]
        print(f"{i:>3} {before[i]:8.2f}px {after[i]:8.2f}px   {ox:8.1f} {oy:8.1f} {LAYOUT.rotations_deg[k]:6.2f}°"
              f"   {nx:8.1f} {ny:8.1f} {fitted.rotations_deg[k]:6.2f}°   {sides[k]:.1f} mm")
    print(f"\nMarker size: plate.yml says {LAYOUT.size_mm:g} mm, the camera sees {fitted.size_mm:.2f} mm "
          f"(each: " + ", ".join(f"{v:.2f}" for v in sides) + ")")
    if abs(fitted.size_mm - LAYOUT.size_mm) > 0.5:
        print("  More than 0.5 mm apart: check the print against its 100 mm scale bar, and the hand-measured centres.")
    if np.ptp(sides) > 0.5:
        print("  The markers differ from each other: one may be printed or cut differently, or not lying flat.")
    moved = np.linalg.norm(fitted.centers_mm - LAYOUT.centers_mm, axis=1)
    worst = LAYOUT.ids[int(np.argmax(moved))]
    print(f"Biggest correction: id {worst}, {moved.max():.1f} mm.")
    if np.mean(list(after.values())) > 1.5:
        print("Still high after the fit: the plate or a marker is probably not flat, or the lens calibration is off.")

    if args.write:
        write_positions(fitted)
        print(f"\nWrote the measured size and positions into {paths.PLATE_FILE.relative_to(paths.REPO)}. "
              "Now run: python -m calibration.set_origin")
    else:
        print("\nTo use these: python -m checks.plate_fit --write   (then calibration.set_origin)")


if __name__ == "__main__":
    main()
