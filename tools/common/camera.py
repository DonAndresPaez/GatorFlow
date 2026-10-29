'''camera.py: where frames come from, and the lens numbers that go with them.

Imported by: calibration/*, checks/*
Run directly: no

Which camera is used, and how, comes from setup/camera.yml. The camera has
a read() that returns a BGR image or None, and release() to clean up. Any
USB/UVC camera works, through OpenCV.
'''

import os
from dataclasses import dataclass

import cv2
import numpy as np

from common import paths


@dataclass
class CameraProfile:
    name: str
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


# ---------------------------------------------------------------- the camera

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


def open_camera(profile=None):
    return Webcam(profile or PROFILE)
