import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracking.track import Smoother, pose_to_td


def test_marker_in_front_of_camera_is_at_negative_z_in_td():
    facing_camera = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    tx, ty, tz, rx, ry, rz = pose_to_td(facing_camera, [0, 0, 500])
    assert (tx, ty, tz) == pytest.approx((0, 0, -500))
    assert (rx, ry, rz) == pytest.approx((0, 0, 0), abs=1e-9)


def test_up_in_the_image_is_up_in_td():
    # OpenCV Y points down, so above the image centre is negative Y.
    assert pose_to_td(np.eye(3), [0, -100, 500])[1] == pytest.approx(100)


def test_smoother_settles_on_target():
    s = Smoother(alpha=0.5)
    s([0, 0, 0, 0, 0, 0])
    for _ in range(30):
        pose = s([100, 0, 0, 0, 60, 0])
    assert pose[0] == pytest.approx(100, abs=1e-3)
    assert pose[4] == pytest.approx(60, abs=1e-3)
