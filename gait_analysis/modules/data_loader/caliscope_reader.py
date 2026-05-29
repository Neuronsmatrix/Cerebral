"""Read caliscope 3D output into the unified pose DataFrame."""
from pathlib import Path

import numpy as np
import pandas as pd

from .landmarks import GAIT_LANDMARKS, MODELS


def list_landmarks(model: str) -> list[str]:
    """Gait-relevant canonical landmarks for a caliscope model."""
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    return list(GAIT_LANDMARKS)


def derive_fps(timestamps) -> float:
    """Frame rate = 1 / median positive inter-frame interval."""
    ts = np.sort(np.asarray(timestamps, dtype=float))
    d = np.diff(ts)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("Cannot derive fps from constant/empty timestamps")
    return float(1.0 / np.median(d))


def _frame_timestamps(frame_time_csv: Path) -> pd.DataFrame:
    """Collapse per-port frame_time to one zero-based timestamp per sync_index."""
    fth = pd.read_csv(frame_time_csv)
    per_sync = fth.groupby("sync_index", as_index=False)["frame_time"].mean()
    per_sync["timestamp"] = per_sync["frame_time"] - per_sync["frame_time"].min()
    return per_sync[["sync_index", "timestamp"]]


def load_caliscope_session(session_dir: str, model: str = "SIMPLE_HOLISTIC") -> pd.DataFrame:
    """Load ``xyz_<model>_labelled.csv`` + real timestamps into a unified DataFrame.

    Returns columns ``frame, timestamp, <landmark>_{x,y,z}``. Sets
    ``df.attrs['fps']`` (derived) and ``df.attrs['model']``.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model {model!r}; expected one of {MODELS}")
    model_dir = Path(session_dir) / model
    labelled = pd.read_csv(model_dir / f"xyz_{model}_labelled.csv")
    times = _frame_timestamps(model_dir / "frame_time_history.csv")

    df = labelled.merge(times, on="sync_index", how="left")
    df = df.sort_values("sync_index").reset_index(drop=True)
    df.insert(0, "frame", range(len(df)))

    coord_cols = [c for c in df.columns
                  if c.endswith(("_x", "_y", "_z")) and c not in ("frame",)]
    df = df[["frame", "timestamp"] + coord_cols]

    df.attrs["fps"] = derive_fps(df["timestamp"].to_numpy())
    df.attrs["model"] = model
    return df
