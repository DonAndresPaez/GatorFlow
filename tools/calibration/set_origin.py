'''set_origin.py: tell the tracker where the world's origin is.

Run: python -m calibration.set_origin         (from tools/)
Makes: calibration/<camera name>/world_origin.yml  (read by KinectReader too)
Needs: a calibrated camera, and the marker plate sitting still on its home spot

Without this, poses are measured from the camera, so the model reads as "1.9 m
away and tilted" because that is how the Kinect hangs. This watches the plate
at its home spot for a few seconds and saves the matrix that makes that spot
read as zero. Afterwards the tracker reports where the plate (and the model
sitting on it) is relative to home, which is what TouchDesigner wants.

Only frames where EVERY marker in the layout is seen are used, so the origin
is always captured from the full plate.

Rerun it whenever the camera moves (even slightly), after a new lens
calibration, and after setup/plate.yml changes. It encodes where the camera
is relative to the plate's home spot.
'''

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from common import paths
from common.camera import PROFILE, check_frame_size, load_intrinsics, open_camera
from common.plate import LAYOUT, OPENCV_TO_TD, MarkerTracker, pose_to_td

FRAMES = 90  # about 3 seconds


def camera_to_world_from_home(R_home, t_home):
    """Given the plate's pose at the home spot, return the matrix that makes
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
    lens = load_intrinsics()
    tracker = MarkerTracker(lens.K, lens.dist)
    camera = open_camera()
    need = len(LAYOUT.ids)
    print(f"Camera '{PROFILE.name}', plate of {need} x {LAYOUT.size_mm:g} mm {LAYOUT.dictionary_name} markers.")
    size_checked = False
    print(f"Plate on the home spot, all {need} markers visible, nothing moving. Collecting {FRAMES} readings...")

    poses, errors = [], []
    while len(poses) < FRAMES:
        frame = camera.read()
        if frame is None:
            continue
        if not size_checked:
            size_checked = True
            warning = check_frame_size(frame, lens)
            if warning:
                print("WARNING:", warning)
        corners, ids = tracker.detect(frame)
        pose = tracker.solve(corners, ids)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        if pose and len(pose.marker_ids) == need:
            poses.append((pose.R, pose.t))
            errors.append(pose.error_px)
            cv2.drawFrameAxes(frame, tracker.K, tracker.dist, pose.rvec, pose.tvec, 150, 3)
            status, color = f"{len(poses)}/{FRAMES}", (0, 255, 0)
        elif pose:
            status, color = f"{len(poses)}/{FRAMES}  waiting: sees {pose.marker_ids}, needs all {need}", (0, 200, 255)
        elif tracker.last_error_px:
            status, color = f"{len(poses)}/{FRAMES}  rejected: fit error {tracker.last_error_px:.1f} px (check setup/plate.yml)", (0, 0, 255)
        else:
            status, color = f"{len(poses)}/{FRAMES}  plate not seen", (0, 0, 255)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.imshow("set origin", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

    if len(poses) < FRAMES // 2:
        print("Not enough readings with the whole plate visible. Are all markers in frame and lit?")
        return

    R, t = average_pose(poses)
    spread = np.std([p for _, p in poses], axis=0)
    print(f"Distance from camera to plate centre: {np.linalg.norm(t):.0f} mm")
    print(f"Noise while still (std dev): {spread.round(2)} mm  <- over ~1 mm means glare, blur, or a bent plate")
    print(f"Fit error: {np.mean(errors):.2f} px average  <- over ~1 px means the layout numbers are off")

    camera_to_world = camera_to_world_from_home(R, t)
    path = paths.origin_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.write("cameraToWorld", camera_to_world)
    fs.release()
    print(f"Wrote {path.relative_to(paths.REPO)}  (read by KinectReader)")
    print("Check (should be six zeros):", [round(v, 2) for v in pose_to_td(R, t, camera_to_world)])


if __name__ == "__main__":
    main()