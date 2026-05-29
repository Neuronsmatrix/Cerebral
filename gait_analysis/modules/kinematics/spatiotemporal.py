"""Spatiotemporal gait parameters from events + heel positions."""
import numpy as np
import pandas as pd


def _heel_xy(df: pd.DataFrame, side: str, vertical: str) -> np.ndarray:
    axes = [a for a in ("x", "y", "z") if a != vertical]
    return df[[f"{side}_heel_{axes[0]}", f"{side}_heel_{axes[1]}"]].to_numpy()


def _travel_axis(positions: np.ndarray) -> np.ndarray:
    """Principal horizontal direction of motion (unit vector) via PCA."""
    centered = positions - np.nanmean(positions, axis=0)
    centered = centered[~np.isnan(centered).any(axis=1)]
    if len(centered) < 2:
        return np.array([1.0, 0.0])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def calc_spatiotemporal(df: pd.DataFrame, events: dict, fps: float,
                        vertical: str = "z") -> dict:
    """Compute cadence, speed, stride/step length, step width, stance/swing %.

    KNOWN LIMITATIONS (Phase 1):
    - ``stance_pct`` / ``swing_pct`` / ``double_support_pct`` derive from the
      upstream toe-off estimator, which currently detects the mid-swing
      toe-height maximum rather than true toe-off. These temporal phase
      parameters are therefore systematically biased until ``detect_gait_events``
      gains a kinematic/velocity TO estimator (validated against Vicon in a
      later phase). Cadence/speed/stride/step length are unaffected.
    - ``step_length_m`` is sensitive to finite tracking artifacts in heel
      position that survive gap-filling (``fill_gaps`` only fills NaN, not
      finite outliers); values well outside ~0.2-0.9 m should be treated as
      suspect.
    """
    timestamps = df["timestamp"].to_numpy()
    left_hs = sorted(events.get("left_HS", []))
    right_hs = sorted(events.get("right_HS", []))
    all_hs = sorted(left_hs + right_hs)

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else np.nan
    n_steps = max(len(all_hs) - 1, 0)
    cadence = (n_steps / duration) * 60.0 if duration and n_steps else np.nan

    heel_left = _heel_xy(df, "left", vertical)
    travel = _travel_axis(heel_left)

    strides = []
    for a, b in zip(left_hs[:-1], left_hs[1:]):
        strides.append(np.linalg.norm(heel_left[b] - heel_left[a]))
    stride_length = float(np.mean(strides)) if strides else np.nan

    speed = (stride_length * cadence / 120.0
             if not np.isnan(stride_length) and not np.isnan(cadence) else np.nan)

    heel_right = _heel_xy(df, "right", vertical)
    perp = np.array([-travel[1], travel[0]])
    step_lengths, step_widths = [], []
    for hs in right_hs:
        if hs < len(heel_right) and hs < len(heel_left):
            vec = heel_right[hs] - heel_left[hs]
            step_lengths.append(abs(np.dot(vec, travel)))
            step_widths.append(abs(np.dot(vec, perp)))
    step_length = float(np.nanmean(step_lengths)) if step_lengths else np.nan
    step_width = float(np.nanmean(step_widths)) if step_widths else np.nan

    left_to = sorted(events.get("left_TO", []))
    stance_fracs = []
    for i in range(len(left_hs) - 1):
        hs0, hs1 = left_hs[i], left_hs[i + 1]
        tos = [t for t in left_to if hs0 < t < hs1]
        if tos:
            stance_fracs.append((tos[0] - hs0) / (hs1 - hs0))
    stance_pct = float(np.mean(stance_fracs) * 100.0) if stance_fracs else np.nan
    swing_pct = 100.0 - stance_pct if not np.isnan(stance_pct) else np.nan
    double_support_pct = max(2 * stance_pct - 100.0, 0.0) if not np.isnan(stance_pct) else np.nan

    return {
        "cadence_steps_per_min": cadence,
        "speed_m_per_s": speed,
        "stride_length_m": stride_length,
        "step_length_m": step_length,
        "step_width_m": step_width,
        "stance_pct": stance_pct,
        "swing_pct": swing_pct,
        "double_support_pct": double_support_pct,
    }
