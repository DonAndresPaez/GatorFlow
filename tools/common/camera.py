'''camera.py: where frames come from, and the lens numbers that go with them.

Imported by: calibration/*, checks/*
Run directly: no

Which camera is used, and how, comes from setup/camera.yml. Every source has
the same two methods, so the callers don't care which is in use: read()
returns a BGR image or None, release() cleans up.

The Kinect v2 is not a webcam: it speaks its own USB protocol, so OpenCV can
never open it whatever index you try. Its frames come through the Kinect SDK
via pykinect2 instead. Any normal USB camera goes through OpenCV.
'''

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np

from common import paths


@dataclass
class CameraProfile:
    name: str
    source: str          # "kinect_v2" or "webcam"
    mirrored: bool
    index: int = 0
    width: int = 0
    height: int = 0
    fps: int = 0
    mjpg: bool = False
    autofocus: bool = False
    focus: int = -1


def load_profile(path=paths.CAMERA_FILE):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"Could not open {path}")

    def number(key, default):
        node = fs.getNode(key)
        return default if node.empty() else node.real()

    profile = CameraProfile(
        name=fs.getNode("name").string(),
        source=fs.getNode("source").string() or "kinect_v2",
        mirrored=bool(number("mirrored", 0)),
        index=int(number("index", 0)),
        width=int(number("width", 0)),
        height=int(number("height", 0)),
        fps=int(number("fps", 0)),
        mjpg=bool(number("mjpg", 0)),
        autofocus=bool(number("autofocus", 0)),
        focus=int(number("focus", -1)),
    )
    fs.release()
    if not profile.name:
        raise ValueError(f"{path} needs a name")
    return profile


PROFILE = load_profile()


# ---------------------------------------------------------------- lens numbers

@dataclass
class Intrinsics:
    K: np.ndarray           # 3x3 camera matrix
    dist: np.ndarray        # distortion coefficients
    image_size: tuple       # (width, height) the calibration was taken at
    error_px: float


def load_intrinsics(name=None):
    """This camera's lens numbers from calibration/<name>/intrinsics.yml."""
    path = paths.intrinsics_file(name)
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(
            f"No lens calibration for camera '{name or PROFILE.name}' ({path}).\n"
            f"Run: python -m calibration.camera_lens")
    K = fs.getNode("cameraMatrix").mat()
    dist = fs.getNode("distortionCoefficients").mat()
    size = (int(fs.getNode("imageWidth").real()), int(fs.getNode("imageHeight").real()))
    error = fs.getNode("reprojectionErrorPx").real()
    fs.release()
    if K is None or dist is None:
        raise ValueError(f"{path} has no cameraMatrix / distortionCoefficients")
    return Intrinsics(K.astype(float), dist.ravel().astype(float), size, error)


def check_frame_size(frame, intrinsics):
    """The lens numbers only hold at the resolution they were measured at.
    Returns a warning string, or None if the frame matches."""
    height, width = frame.shape[:2]
    if intrinsics.image_size[0] and (width, height) != tuple(intrinsics.image_size):
        return (f"Frames are {width}x{height} but the calibration was taken at "
                f"{intrinsics.image_size[0]}x{intrinsics.image_size[1]}. "
                f"Rerun python -m calibration.camera_lens")
    return None


# ---------------------------------------------------------------- sources

class Webcam:
    def __init__(self, profile):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(profile.index, backend)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open webcam index {profile.index}. "
                               f"Try another index in setup/camera.yml.")
        if profile.mjpg:
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        if profile.width and profile.height:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, profile.width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, profile.height)
        if profile.fps:
            self.capture.set(cv2.CAP_PROP_FPS, profile.fps)
        self.capture.set(cv2.CAP_PROP_AUTOFOCUS, 1 if profile.autofocus else 0)
        if profile.focus >= 0:
            self.capture.set(cv2.CAP_PROP_FOCUS, profile.focus)
        self.mirrored = profile.mirrored

        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Webcam {profile.index}: {width}x{height} @ {self.capture.get(cv2.CAP_PROP_FPS):.0f} fps")
        if profile.width and (width, height) != (profile.width, profile.height):
            print(f"WARNING: asked for {profile.width}x{profile.height}, the camera gave {width}x{height}.")

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            return None
        return cv2.flip(frame, 1) if self.mirrored else frame

    def release(self):
        self.capture.release()


class KinectV2:
    """Color stream from a Kinect for Xbox One (always 1920x1080).

    Needs: Kinect for Windows SDK 2.0, a USB 3 port, 64-bit Python, and
        pip install pykinect2 comtypes
    If importing pykinect2 crashes on an assert, run:
        python -m common.patch_pykinect2
    """

    def __init__(self, profile):
        try:
            from pykinect2 import PyKinectRuntime, PyKinectV2
        except AssertionError as error:
            raise RuntimeError("pykinect2 needs patching for this Python. "
                               "Run: python -m common.patch_pykinect2") from error
        except ImportError as error:
            if "Wrong version" in str(error) or "_check_version" in str(error):
                raise RuntimeError("pykinect2 clashes with this comtypes version. "
                                   "Run: python -m common.patch_pykinect2") from error
            raise RuntimeError("pykinect2 is not installed. Run: pip install pykinect2 comtypes") from error

        self.mirrored = profile.mirrored
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
        print("  programs using the sensor (KinectReader.exe?), and confirm Kinect Studio sees it.")

    def read(self):
        if not self.kinect.has_new_color_frame():
            time.sleep(0.005)
            return None
        bgra = self.kinect.get_last_color_frame().reshape((self.height, self.width, 4)).astype(np.uint8)
        frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        # The Kinect hands over its color frame mirrored. ArUco reads a mirrored
        # marker as a different code entirely, and solvePnP would return a
        # left-handed pose, so the picture gets flipped back here.
        return cv2.flip(frame, 1) if self.mirrored else frame

    def release(self):
        self.kinect.close()


def open_camera(profile=None):
    profile = profile or PROFILE
    if profile.source == "kinect_v2":
        return KinectV2(profile)
    if profile.source == "webcam":
        return Webcam(profile)
    raise ValueError(f"setup/camera.yml: unknown source '{profile.source}' (use kinect_v2 or webcam)")
