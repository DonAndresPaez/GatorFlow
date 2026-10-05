'''live_view.py: live check that the marker plate is detected, nothing else.

Run: python -m checks.live_view      (from tools/)
Keys: q = quit, s = save the current frame to tools/output/detection_snapshot.png

Shows the camera feed with each marker outlined and the PLATE's axes drawn at
its centre, and on screen:
  - LOCK / lost, and the detection rate over the last ~2 seconds
  - which of the layout's markers were used this frame
  - fit error: how far the corners land from where setup/plate.yml says
    they should be (under ~1 px good; high means setup/plate.yml is off)
  - average marker size in pixels, and distance to the plate centre
  - the pose relative to the home spot, if set_origin has been run

It sends no OSC. The live tracker is KinectReader; this only answers "does the
camera see the plate, and how well". It uses the same solve as set_origin
(common/plate.py), so what it shows is what the origin capture would see.

Close KinectReader first if the feed stays black.
'''

import time
from collections import deque

import cv2
import numpy as np

from common import paths
from common.camera import PROFILE, check_frame_size, load_intrinsics, open_camera
from common.plate import LAYOUT, MAX_REPROJECTION_ERROR_PX, MarkerTracker, pose_to_td

WINDOW = "GatorFlow detection test"
HISTORY = 60  # frames used for the detection rate (~2 s at 30 fps)


def mean_marker_pixels(corners, ids):
    """Average side length in pixels of the layout markers seen, or None."""
    if ids is None:
        return None
    sizes = [np.mean([np.linalg.norm(c.reshape(4, 2)[k] - c.reshape(4, 2)[(k + 1) % 4]) for k in range(4)])
             for c, i in zip(corners, np.ravel(ids)) if int(i) in LAYOUT.ids]
    return float(np.mean(sizes)) if sizes else None


def put(frame, text, row, color=(255, 255, 255)):
    y = 40 + row * 38
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)


def main():
    lens = load_intrinsics()
    tracker = MarkerTracker(lens.K, lens.dist)
    has_origin = paths.origin_file().exists()
    print(f"Camera '{PROFILE.name}' ({PROFILE.source}).")
    print(f"Looking for {LAYOUT.dictionary_name}, ids {LAYOUT.ids}, {LAYOUT.size_mm:g} mm, from setup/plate.yml.")
    print("Origin:", "loaded, poses are relative to home" if has_origin else "not set, poses are relative to the camera")
    print("q = quit, s = save snapshot")

    camera = open_camera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    hits = deque(maxlen=HISTORY)
    last_print = time.time()
    size_checked = False

    while True:
        frame = camera.read()
        if frame is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        if not size_checked:
            size_checked = True
            warning = check_frame_size(frame, lens)
            if warning:
                print("WARNING:", warning)

        corners, ids = tracker.detect(frame)
        pose = tracker.solve(corners, ids)
        size_px = mean_marker_pixels(corners, ids)
        hits.append(pose is not None)
        rate = 100 * sum(hits) / len(hits)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        if pose:
            cv2.drawFrameAxes(frame, tracker.K, tracker.dist, pose.rvec, pose.tvec, 150, 3)
            td = pose_to_td(pose.R, pose.t)
            n = len(pose.marker_ids)
            put(frame, f"LOCK   detected {rate:3.0f}% of recent frames", 0, (0, 255, 0))
            put(frame, f"markers {pose.marker_ids} ({n}/{len(LAYOUT.ids)})   fit error {pose.error_px:.2f} px",
                1, (0, 255, 0) if n == len(LAYOUT.ids) and pose.error_px < 1.0 else (0, 200, 255))
            put(frame, f"marker size {size_px:4.0f} px   plate centre {np.linalg.norm(pose.t):5.0f} mm away", 2)
            label = "from home" if has_origin else "from camera"
            put(frame, f"pos {td[0]:7.1f} {td[1]:7.1f} {td[2]:7.1f} mm ({label})", 3)
            put(frame, f"rot {td[3]:7.1f} {td[4]:7.1f} {td[5]:7.1f} deg", 4)
        else:
            put(frame, f"lost   detected {rate:3.0f}% of recent frames", 0, (0, 0, 255))
            if tracker.last_error_px:
                put(frame, f"rejected: fit error {tracker.last_error_px:.1f} px > {MAX_REPROJECTION_ERROR_PX:.0f}"
                           f"  -> measure setup/plate.yml again", 1, (0, 0, 255))
            elif ids is not None:
                put(frame, f"seeing ids {np.ravel(ids).tolist()}, none of {LAYOUT.ids}", 1, (0, 200, 255))

        now = time.time()
        if now - last_print >= 1.0:
            if pose:
                print(f"LOCK {rate:3.0f}%  markers {pose.marker_ids}  err {pose.error_px:4.2f} px  "
                      f"pos {td[0]:7.1f} {td[1]:7.1f} {td[2]:7.1f}  rot {td[3]:6.1f} {td[4]:6.1f} {td[5]:6.1f}")
            elif tracker.last_error_px:
                print(f"lost {rate:3.0f}%  rejected, fit error {tracker.last_error_px:.1f} px")
            else:
                print(f"lost {rate:3.0f}%")
            last_print = now

        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            snapshot = paths.output_file("detection_snapshot.png")
            cv2.imwrite(str(snapshot), frame)
            print(f"Saved {snapshot}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()