"""GL-free skeleton geometry: connectivity + per-frame point/segment extraction."""
import pandas as pd

# Drawable skeleton connectivity (canonical lowercase landmark names).
SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
]


def frame_points(df, frame_idx):
    """Return {landmark: (x, y, z)} for one frame, omitting missing/NaN landmarks."""
    row = df.iloc[frame_idx]
    points = {}
    for col in df.columns:
        if not col.endswith("_x"):
            continue
        name = col[:-2]
        xyz = (row.get(f"{name}_x"), row.get(f"{name}_y"), row.get(f"{name}_z"))
        if any(v is None for v in xyz) or any(pd.isna(v) for v in xyz):
            continue
        points[name] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    return points


def segment_lines(points, edges=SKELETON_EDGES):
    """Return [((x,y,z),(x,y,z)), ...] for edges whose endpoints both exist in points."""
    lines = []
    for a, b in edges:
        if a in points and b in points:
            lines.append((points[a], points[b]))
    return lines
