"""Coordinate-system registration (Umeyama) and temporal sync for Vicon comparison.

Ported and cleaned from the VvsC.py baseline (compute_similarity_transform,
estimate_time_shift_by_xcorr, detect_event_index).
"""
import numpy as np


def estimate_rigid_transform(src: np.ndarray, dst: np.ndarray):
    """Umeyama (1991) similarity: find (R, T, s) with dst ≈ s * (src @ R.T) + T.

    Reflection-free (det(R) = +1), uniform scale. src/dst are (N, 3).
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    H = s0.T @ d0
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_s = (s0 ** 2).sum()
    s = S.sum() / var_s if var_s > 0 else 1.0
    T = mu_d - s * (R @ mu_s)
    return R, T, s


def apply_transform(points: np.ndarray, R: np.ndarray, T: np.ndarray, s: float) -> np.ndarray:
    """Apply (R, T, s): s * (points @ R.T) + T."""
    return s * (np.asarray(points, float) @ R.T) + T
