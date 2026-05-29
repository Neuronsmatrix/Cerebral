import numpy as np
import pytest
from modules.data_loader.caliscope_reader import list_landmarks, derive_fps
from modules.data_loader.landmarks import GAIT_LANDMARKS


def test_list_landmarks_returns_gait_set_for_known_model():
    lm = list_landmarks("SIMPLE_HOLISTIC")
    assert "left_knee" in lm and "right_foot_index" in lm
    assert lm == GAIT_LANDMARKS


def test_list_landmarks_rejects_unknown_model():
    with pytest.raises(ValueError):
        list_landmarks("NOT_A_MODEL")


def test_derive_fps_from_even_timestamps():
    ts = np.arange(0, 1.0, 1 / 20.0)  # 20 fps
    assert derive_fps(ts) == pytest.approx(20.0, abs=0.01)


def test_derive_fps_robust_to_a_dropped_frame():
    ts = np.array([0.0, 0.05, 0.10, 0.20, 0.25])  # one gap at 0.15
    assert derive_fps(ts) == pytest.approx(20.0, abs=0.01)
