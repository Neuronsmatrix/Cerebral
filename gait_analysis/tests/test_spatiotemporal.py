import numpy as np
import pandas as pd
import pytest

from modules.kinematics.spatiotemporal import calc_spatiotemporal


def _constant_velocity_walk(fps=50.0):
    t = np.arange(0, 4.0, 1 / fps)
    x = 1.0 * t  # +x at 1 m/s
    df = pd.DataFrame({
        "frame": range(len(t)), "timestamp": t,
        "left_heel_x": x, "left_heel_y": np.zeros_like(t), "left_heel_z": np.zeros_like(t),
        "right_heel_x": x, "right_heel_y": np.full_like(t, 0.2), "right_heel_z": np.zeros_like(t),
    })
    return df, fps


def test_cadence_and_speed_are_positive_and_reasonable():
    df, fps = _constant_velocity_walk()
    events = {
        "left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
        "right_HS": [25, 75, 125], "right_TO": [55, 105, 155],
    }
    out = calc_spatiotemporal(df, events, fps=fps)
    assert out["cadence_steps_per_min"] > 0
    assert out["speed_m_per_s"] > 0
    assert out["stride_length_m"] == pytest.approx(1.0, abs=0.2)


def test_stance_swing_sum_to_about_100():
    df, fps = _constant_velocity_walk()
    events = {
        "left_HS": [0, 50, 100, 150], "left_TO": [30, 80, 130],
        "right_HS": [25, 75, 125], "right_TO": [55, 105, 155],
    }
    out = calc_spatiotemporal(df, events, fps=fps)
    assert out["stance_pct"] + out["swing_pct"] == pytest.approx(100.0, abs=1.0)
