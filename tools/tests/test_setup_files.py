'''test_setup_files.py - checks the hand-edited files in setup/ and models/ load.

Run: pytest

These are the files people edit when the rig changes, so a typo there should
fail here, at the desk, rather than in the lab.
'''

import cv2
import numpy as np
import pytest

from common import paths
from common.camera import load_profile
from common.plate import load_layout


def test_camera_profile_loads():
    profile = load_profile()
    assert profile.name
    assert profile.source in ("kinect_v2", "webcam")


def test_plate_loads_and_markers_do_not_overlap():
    layout = load_layout()
    assert hasattr(cv2.aruco, layout.dictionary_name)
    assert len(set(layout.ids)) == len(layout.ids), "a marker id is listed twice"
    for a in range(len(layout.ids)):
        for b in range(a + 1, len(layout.ids)):
            gap = np.linalg.norm(layout.centers_mm[a] - layout.centers_mm[b])
            assert gap > layout.size_mm, f"markers {layout.ids[a]} and {layout.ids[b]} overlap"


def test_plate_format(tmp_path):
    plate = tmp_path / "plate.yml"
    plate.write_text('%YAML:1.0\n---\ndictionary: "DICT_5X5_50"\nmarkerSizeMm: 30.5   # measured\n'
                     "markers:\n   - { id: 7, xMm: 10., yMm: -20. }\n"
                     "   - { id: 9, xMm: -10., yMm: 20., rotationDeg: 90. }\n")
    layout = load_layout(plate)
    assert layout.dictionary_name == "DICT_5X5_50"
    assert layout.size_mm == pytest.approx(30.5)
    assert layout.ids == [7, 9]
    assert layout.rotations_deg.tolist() == [0.0, 90.0]


def test_unknown_dictionary_is_rejected(tmp_path):
    plate = tmp_path / "plate.yml"
    plate.write_text('%YAML:1.0\n---\ndictionary: "DICT_NOPE"\nmarkerSizeMm: 30.\n'
                     "markers:\n   - { id: 0, xMm: 0., yMm: 0. }\n")
    with pytest.raises(ValueError):
        load_layout(plate)


def test_active_model_exists():
    model = paths.active_model()
    assert model["stl"].exists(), f"{model['stl']} is missing"
    assert len(model["mountOffsetMm"]) == 3
