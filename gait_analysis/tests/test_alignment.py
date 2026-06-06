import numpy as np

from modules.comparison.alignment import apply_transform, estimate_rigid_transform


def _rot_z(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0, 0, 1.0]])


def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(42)
    src = rng.normal(size=(50, 3))
    R_true, s_true, T_true = _rot_z(30.0), 2.5, np.array([1.0, -2.0, 0.5])
    dst = s_true * (src @ R_true.T) + T_true

    R, T, s = estimate_rigid_transform(src, dst)
    assert np.isclose(s, s_true, atol=1e-6)
    assert np.allclose(R, R_true, atol=1e-6)
    assert np.allclose(T, T_true, atol=1e-6)
    assert np.linalg.det(R) > 0          # reflection-free
    assert np.allclose(apply_transform(src, R, T, s), dst, atol=1e-6)
