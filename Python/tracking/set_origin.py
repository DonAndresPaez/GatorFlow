'''set_origin.py: tell the tracker where the world's origin is.

Run: python -m tracking.set_origin
Makes: tracking/world_origin.json and shared/world_origin.yml
Needs: a calibrated camera, and the marker sitting still on the home spot

Without this, poses are measured from the camera, so the model reads as "1.5 m
away and tilted 30 degrees" because that is how the Kinect hangs. This watches
the marker at its home spot for two seconds and saves the matrix that makes
that spot read as zero. Afterwards the tracker reports where the model is
relative to home, which is what TouchDesigner wants.

Rerun it whenever the Kinect moves, even slightly. It encodes where the
sensor is.
'''

import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tracking.camera import open_camera
from tracking.marker import (CAMERA_FILE, OPENCV_TO_TD, ORIGIN_FILE, SHARED_DIR,
                             MarkerTracker, pose_to_td)

FRAMES = 60  # about 2 seconds
YAML_FILE = SHARED_DIR / "world_origin.yml"   # what KinectReader (C++) reads


def camera_to_world_from_home(R_home, t_home):
    """Given the marker's pose at the home spot, return the matrix that makes
    that pose come out as all zeros."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R_home, np.ravel(t_home)
    return np.linalg.inv(OPENCV_TO_TD @ T)


def average_pose(poses):
    """Mean position, and mean rotation via quaternions (so angles near the
    wrap-around don't average to nonsense)."""
    positions = np.array([t for _, t in poses])
    quats = np.array([Rotation.from_matrix(R).as_quat() for R, _ in poses])
    quats[np.einsum("ij,j->i", quats, quats[0]) < 0] *= -1  # same hemisphere as the first
    mean_quat = quats.mean(0)
    return Rotation.from_quat(mean_quat / np.linalg.norm(mean_quat)).as_matrix(), positions.mean(0)


def main():
    cam = json.loads(CAMERA_FILE.read_text())
    tracker = MarkerTracker(cam["camera_matrix"], cam["dist_coeffs"])
    camera = open_camera()
    print(f"Hold the marker still at the home spot. Collecting {FRAMES} readings...")

    poses = []
    while len(poses) < FRAMES:
        frame = camera.read()
        if frame is None:
            continue
        found = tracker.find(frame)
        if found:
            R, t, rvec, tvec = found
            poses.append((R, t))
            cv2.drawFrameAxes(frame, tracker.K, tracker.dist, rvec, tvec, 80)
        cv2.putText(frame, f"{len(poses)}/{FRAMES}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("set origin", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

    if len(poses) < FRAMES // 2:
        print("Not enough readings. Is the marker visible and lit?")
        return

    R, t = average_pose(poses)
    spread = np.std([p for _, p in poses], axis=0)
    print(f"Distance from camera: {np.linalg.norm(t):.0f} mm")
    print(f"Noise while still (std dev): {spread.round(1)} mm  <- over ~2 mm means glare, blur, or a bent marker")

    camera_to_world = camera_to_world_from_home(R, t)
    ORIGIN_FILE.write_text(json.dumps({"camera_to_world": camera_to_world.tolist()}, indent=2))
    print(f"Wrote {ORIGIN_FILE}")

    YAML_FILE.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(YAML_FILE), cv2.FILE_STORAGE_WRITE)
    fs.write("cameraToWorld", camera_to_world)
    fs.release()
    print(f"Wrote {YAML_FILE}  (used by KinectReader)")
    print("Check (should be six zeros):", [round(v, 2) for v in pose_to_td(R, t, camera_to_world)])


if __name__ == "__main__":
    main()
