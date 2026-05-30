import numpy as np
import pandas as pd

from modules.data_loader.landmarks import GAIT_LANDMARKS
from modules.visualization.skeleton_3d import SKELETON_EDGES, frame_points, segment_lines


def test_frame_points_extracts_xyz_and_skips_nan():
    df = pd.DataFrame({
        "left_hip_x": [0.0], "left_hip_y": [1.0], "left_hip_z": [2.0],
        "left_knee_x": [np.nan], "left_knee_y": [1.0], "left_knee_z": [0.0],
    })
    pts = frame_points(df, 0)
    assert pts["left_hip"] == (0.0, 1.0, 2.0)
    assert "left_knee" not in pts          # NaN endpoint dropped


def test_segment_lines_skips_edges_with_missing_endpoint():
    points = {"left_hip": (0, 0, 0), "left_knee": (0, 0, 1)}
    lines = segment_lines(points, edges=[("left_hip", "left_knee"),
                                         ("left_knee", "left_ankle")])
    assert lines == [((0, 0, 0), (0, 0, 1))]


def test_skeleton_edges_reference_known_landmarks():
    names = set(GAIT_LANDMARKS)
    for a, b in SKELETON_EDGES:
        assert a in names and b in names
