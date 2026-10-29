'''test_marker.py - checks the axis conversion in common/plate.py.

Run: pytest

set_origin and the tracker both use this transform, so if these tests fail
the origin capture is producing a matrix the tracker reads differently.
'''

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from common.plate import pose_to_td


def test_marker_in_front_of_camera_is_at_negative_z_in_td():
    """Checked here because it's what set_origin uses to work out the world matrix."""
    facing_camera = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    tx, ty, tz, rx, ry, rz = pose_to_td(facing_camera, [0, 0, 500], np.eye(4))
    assert (tx, ty, tz) == pytest.approx((0, 0, -500))
    assert (rx, ry, rz) == pytest.approx((0, 0, 0), abs=1e-9)


def test_up_in_the_image_is_up_in_td():
    # OpenCV Y points down, so above the image centre is negative Y.
    assert pose_to_td(np.eye(3), [0, -100, 500], np.eye(4))[1] == pytest.approx(100)
