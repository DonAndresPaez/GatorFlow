'''test_marker.py - checks the axis conversion in marker.py.

Run: pytest

This is the same transform KinectReader applies in poseToWorld(). The two
have to agree, because set_origin uses the Python one to produce the matrix
the C++ one consumes. If these tests fail, the origin capture is producing a
matrix that means something different on the other side.
'''

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracking.marker import pose_to_td


def test_marker_in_front_of_camera_is_at_negative_z_in_td():
    """Same transform KinectReader does in poseToWorld(), checked here because
    it's what set_origin uses to work out the world matrix."""
    facing_camera = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    tx, ty, tz, rx, ry, rz = pose_to_td(facing_camera, [0, 0, 500])
    assert (tx, ty, tz) == pytest.approx((0, 0, -500))
    assert (rx, ry, rz) == pytest.approx((0, 0, 0), abs=1e-9)


def test_up_in_the_image_is_up_in_td():
    # OpenCV Y points down, so above the image centre is negative Y.
    assert pose_to_td(np.eye(3), [0, -100, 500])[1] == pytest.approx(100)
