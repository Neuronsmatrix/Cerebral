"""Draw the gait skeleton onto raw camera videos from caliscope 2D detections (cv2)."""
from pathlib import Path

import cv2
import pandas as pd

from modules.visualization.skeleton_3d import SKELETON_EDGES

# MediaPipe Pose landmark index -> canonical gait landmark name (the 12 skeleton joints).
POSE_POINT_IDS = {
    11: "left_shoulder", 12: "right_shoulder",
    23: "left_hip", 24: "right_hip",
    25: "left_knee", 26: "right_knee",
    27: "left_ankle", 28: "right_ankle",
    29: "left_heel", 30: "right_heel",
    31: "left_foot_index", 32: "right_foot_index",
}

_JOINT_COLOR = (0, 0, 255)   # red (BGR)
_BONE_COLOR = (255, 255, 255)   # white


def load_xy(session_dir, model):
    """Read the per-camera 2D detections CSV for a model."""
    path = Path(session_dir) / model / f"xy_{model}.csv"
    if not path.exists():
        raise FileNotFoundError(f"2D detections not found: {path}")
    return pd.read_csv(path)


def frame_marks(xy_df, port, frame_index):
    """Return {gait_landmark: (x, y)} for one camera frame; missing/NaN landmarks omitted."""
    rows = xy_df[(xy_df["port"] == port) & (xy_df["frame_index"] == frame_index)]
    marks = {}
    for _, r in rows.iterrows():
        name = POSE_POINT_IDS.get(int(r["point_id"]))
        if name is None:
            continue
        x, y = r["img_loc_x"], r["img_loc_y"]
        if pd.isna(x) or pd.isna(y):
            continue
        marks[name] = (int(round(x)), int(round(y)))
    return marks


def draw_overlay(frame, marks, edges=SKELETON_EDGES):
    """Draw bones (lines) + joints (filled circles) onto a BGR frame in place; return it."""
    for a, b in edges:
        if a in marks and b in marks:
            cv2.line(frame, marks[a], marks[b], _BONE_COLOR, 3)
    for (x, y) in marks.values():
        cv2.circle(frame, (x, y), 7, _JOINT_COLOR, -1)
        cv2.circle(frame, (x, y), 7, _BONE_COLOR, 1)
    return frame
