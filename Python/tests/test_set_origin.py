import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tracking.set_origin import average_pose, camera_to_world_from_home
from tracking.track import pose_to_td


def test_home_pose_becomes_all_zeros():
    """Whatever angle the camera is mounted at, the marker at its home spot
    must report position 0 and rotation 0."""
    R_home = Rotation.from_euler("xyz", [200, 15, -30], degrees=True).as_matrix()
    t_home = np.array([120.0, -80.0, 1500.0])
    camera_to_world = camera_to_world_from_home(R_home, t_home)
    assert pose_to_td(R_home, t_home, camera_to_world) == pytest.approx([0] * 6, abs=1e-9)


def test_moving_away_from_home_shows_up_as_distance():
    R_home = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    t_home = np.array([0.0, 0.0, 1500.0])
    camera_to_world = camera_to_world_from_home(R_home, t_home)
    # Same orientation, 100 mm to the camera's right -> +100 on X in TD.
    moved = pose_to_td(R_home, t_home + [100, 0, 0], camera_to_world)
    assert moved[0] == pytest.approx(100)
    assert moved[1:3] == pytest.approx([0, 0], abs=1e-9)


def test_average_pose_smooths_noise():
    rng = np.random.default_rng(0)
    R = Rotation.from_euler("y", 30, degrees=True).as_matrix()
    poses = [(R, np.array([10.0, 20.0, 500.0]) + rng.normal(0, 1, 3)) for _ in range(200)]
    R_avg, t_avg = average_pose(poses)
    assert t_avg == pytest.approx([10, 20, 500], abs=0.5)
    assert np.allclose(R_avg, R)
