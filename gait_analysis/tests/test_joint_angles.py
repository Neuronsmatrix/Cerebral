import numpy as np
import pytest

from modules.kinematics.joint_angles import calc_angle_3d


def test_calc_angle_90_degrees():
    p1 = np.array([1.0, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 1.0, 0.0])
    assert calc_angle_3d(p1, vertex, p2) == pytest.approx(90.0, abs=1e-6)


def test_calc_angle_180_degrees_collinear():
    assert calc_angle_3d(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]),
                         np.array([-1.0, 0, 0])) == pytest.approx(180.0, abs=1e-6)


def test_calc_angle_45_degrees():
    assert calc_angle_3d(np.array([1.0, 0, 0]), np.array([0.0, 0, 0]),
                         np.array([1.0, 1.0, 0])) == pytest.approx(45.0, abs=1e-6)


def test_calc_angle_nan_when_point_missing():
    out = calc_angle_3d(np.array([np.nan, 0, 0]), np.array([0.0, 0, 0]),
                        np.array([0.0, 1, 0]))
    assert np.isnan(out)
