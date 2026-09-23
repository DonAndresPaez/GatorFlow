"""Where frames come from.

The Kinect v2 is not a webcam. It speaks its own USB protocol, so OpenCV can
never see it no matter what index you try. Frames come through the Kinect SDK
instead, via pykinect2.

Every source here has the same two methods, so the rest of the code doesn't
care which one is in use:
    read()     -> BGR image, or None if no new frame yet
    release()  -> clean up
"""

import time

import cv2
import numpy as np

# "kinect_v2" for the real sensor, "webcam" for a laptop camera while testing.
SOURCE = "kinect_v2"
WEBCAM_INDEX = 0

# The Kinect hands over its color frame mirrored, like looking in a mirror.
# ArUco reads a mirrored marker as a different code entirely, and solvePnP
# would return a left-handed pose, so the picture gets flipped back here.
KINECT_MIRRORED = True


class Webcam:
    def __init__(self, index=WEBCAM_INDEX):
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}. Run: python -m tracking.list_cameras")

    def read(self):
        ok, frame = self.capture.read()
        return frame if ok else None

    def release(self):
        self.capture.release()


class KinectV2:
    """Color stream from a Kinect for Xbox One.

    Needs: Kinect for Windows SDK 2.0, a USB 3 port, 64-bit Python, and
        pip install pykinect2 comtypes
    If importing pykinect2 crashes on an assert, run:
        python -m tracking.patch_pykinect2
    """

    def __init__(self):
        try:
            from pykinect2 import PyKinectRuntime, PyKinectV2
        except AssertionError as error:
            raise RuntimeError("pykinect2 needs patching for this Python. "
                               "Run: python -m tracking.patch_pykinect2") from error
        except ImportError as error:
            if "Wrong version" in str(error) or "_check_version" in str(error):
                raise RuntimeError("pykinect2 clashes with this comtypes version. "
                                   "Run: python -m tracking.patch_pykinect2") from error
            raise RuntimeError("pykinect2 is not installed. Run: pip install pykinect2 comtypes") from error

        self.kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
        self.width = self.kinect.color_frame_desc.Width      # 1920
        self.height = self.kinect.color_frame_desc.Height    # 1080

        # The sensor can take a while to start streaming, especially right after
        # another program let go of it. Wait, but don't give up: read() returns
        # None until frames arrive, and every script here copes with that.
        for attempt in range(300):  # up to 15 s
            if self.kinect.has_new_color_frame():
                print(f"Kinect streaming after {attempt * 0.05:.1f}s")
                return
            time.sleep(0.05)
        print("WARNING: Kinect opened but hasn't sent a frame in 15 s. Carrying on anyway.")
        print("  If nothing appears: check the power brick, use a USB 3 port, close other")
        print("  programs using the sensor, and confirm Kinect Studio sees it.")

    def read(self):
        if not self.kinect.has_new_color_frame():
            time.sleep(0.005)
            return None
        bgra = self.kinect.get_last_color_frame().reshape((self.height, self.width, 4)).astype(np.uint8)
        frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return cv2.flip(frame, 1) if KINECT_MIRRORED else frame

    def release(self):
        self.kinect.close()


def open_camera(source=None):
    source = source or SOURCE
    if source == "kinect_v2":
        return KinectV2()
    if source == "webcam":
        return Webcam()
    raise ValueError(f"unknown SOURCE: {source}")
