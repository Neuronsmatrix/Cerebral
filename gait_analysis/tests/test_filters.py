import numpy as np
import pandas as pd
import pytest
from modules.kinematics.filters import fill_gaps


def test_fill_gaps_fills_short_interior_gap_on_linear_ramp():
    # Cubic interpolation through collinear points reproduces the line exactly.
    df = pd.DataFrame({"left_heel_z": [0.0, 2.0, 4.0, np.nan, 8.0, 10.0]})
    out = fill_gaps(df, max_gap_frames=10)
    assert out["left_heel_z"].isna().sum() == 0
    assert out["left_heel_z"].iloc[3] == pytest.approx(6.0)


def test_fill_gaps_leaves_long_gap_untouched():
    col = [0.0, 1.0, np.nan, np.nan, np.nan, 5.0, 6.0]  # 3-long gap
    df = pd.DataFrame({"left_heel_z": col})
    out = fill_gaps(df, max_gap_frames=2)  # 3 > 2 -> stays fully NaN
    assert out["left_heel_z"].iloc[2:5].isna().all()


def test_fill_gaps_ignores_non_coordinate_columns():
    # left_hip_x is the line y = x + 1 with a one-frame hole at index 1.
    df = pd.DataFrame({"frame": [0, 1, 2, 3, 4],
                       "timestamp": [0.0, 0.1, 0.2, 0.3, 0.4],
                       "left_hip_x": [1.0, np.nan, 3.0, 4.0, 5.0]})
    out = fill_gaps(df, max_gap_frames=5)
    assert list(out["frame"]) == [0, 1, 2, 3, 4]
    assert out["left_hip_x"].iloc[1] == pytest.approx(2.0)
