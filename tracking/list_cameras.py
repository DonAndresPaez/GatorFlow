"""Find out which camera index is which.

    python -m tracking.list_cameras

Tries indexes 0-5 with both Windows camera backends, prints what each one
gives, and saves a snapshot as camera_<index>_<backend>.png. Open the PNGs:
the one showing what the Kinect is pointed at is the index you want in
CAMERA_INDEX.
"""

import cv2

BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
MAX_INDEX = 5


def main():
    found = []
    for index in range(MAX_INDEX + 1):
        for name, backend in BACKENDS:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue
            ok, frame = capture.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
                filename = f"camera_{index}_{name}.png"
                cv2.imwrite(filename, frame)
                mean = frame.mean()
                note = "  (all black - device opened but sends no picture)" if mean < 2 else ""
                print(f"index {index} [{name}]: {width}x{height}, brightness {mean:.0f} -> {filename}{note}")
                found.append((index, name))
            else:
                print(f"index {index} [{name}]: opens but no frame")
            capture.release()

    if not found:
        print("\nNo camera opened at all. The Kinect is probably not a plain webcam on this machine.")
    else:
        print(f"\n{len(found)} working. Look at the PNGs and use the index whose picture comes from the Kinect.")
        print("If the only pictures are your laptop webcam or a virtual camera, the Kinect needs its SDK instead.")


if __name__ == "__main__":
    main()
