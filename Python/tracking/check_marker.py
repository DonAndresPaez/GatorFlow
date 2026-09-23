'''check_marker.py: this works out why the marker is not being detected.

Run: python -m tracking.check_marker
Makes: check_frame.png and check_frame_annotated.png

Grabs one frame and reports how bright it is, how sharp it is, how many square
shapes the detector considered, and what it decodes in every ArUco dictionary,
not just ours. The saved images are worth looking at, or sending to someone.

This is what found the Kinect mirroring: a mirrored DICT_4X4_50 id 0 decodes
as DICT_4X4_1000 id 871, so a decode in the wrong dictionary means the image
is flipped, not that the marker is wrong.

Mostly used for debugging, but it is also a good check that the camera is working at all.
'''

import cv2
import numpy as np

from tracking.camera import open_camera
from tracking.marker import ARUCO_DICT, MARKER_ID

WARMUP_FRAMES = 15  # let auto-exposure settle


def grab_frame():
    camera = open_camera()
    frame, kept = None, 0
    for _ in range(2000):  # the Kinect returns None between frames
        got = camera.read()
        if got is None:
            continue
        frame, kept = got, kept + 1
        if kept >= WARMUP_FRAMES:
            break
    camera.release()
    return frame


def main():
    frame = grab_frame()
    if frame is None:
        print("No frame from the camera at all. If this is the Kinect: close any other")
        print("program using it (another track.py window?), unplug and replug it, and retry.")
        return

    cv2.imwrite("check_frame.png", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    brightness = gray.mean()
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"frame: {width}x{height}")
    print(f"brightness: {brightness:.0f} / 255   (under 40 = too dark, over 220 = blown out)")
    print(f"sharpness: {sharpness:.0f}          (under 100 = blurry or out of focus)")

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    ours = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), params)
    corners, ids, rejected = ours.detectMarkers(gray)
    print(f"square candidates the detector looked at: {len(rejected)}"
          f"   (0 means it isn't seeing the marker's outline at all)")

    if ids is not None:
        print(f"\nFOUND with our settings: ids {ids.ravel().tolist()}")
        for c, i in zip(corners, ids.ravel()):
            pts = c.reshape(4, 2)
            size = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2]))
            note = "  <- the one track.py wants" if i == MARKER_ID else ""
            print(f"  id {i}: {size:.0f} px across{note}")
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    else:
        print("\nNot found with our dictionary. Trying every other one...")
        names = [n for n in dir(cv2.aruco) if n.startswith("DICT_")]
        any_hit = False
        for name in names:
            d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            c, i, _ = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
            if i is not None:
                print(f"  {name}: ids {i.ravel().tolist()}")
                any_hit = True
        if not any_hit:
            print("  nothing in any dictionary - it's an image problem, not a dictionary problem")

    cv2.imwrite("check_frame_annotated.png", frame)
    print("\nSaved check_frame.png and check_frame_annotated.png")


if __name__ == "__main__":
    main()
