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
