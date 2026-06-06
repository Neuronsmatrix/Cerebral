import numpy as np
import pytest

from modules.comparison.metrics import calc_icc, calc_mae, calc_pearson, calc_rmse


def test_identity_arrays_give_perfect_scores():
    a = np.linspace(0, 60, 101)
    assert calc_rmse(a, a) == pytest.approx(0.0, abs=1e-12)
    assert calc_mae(a, a) == pytest.approx(0.0, abs=1e-12)
    assert calc_pearson(a, a) == pytest.approx(1.0, abs=1e-9)
    assert calc_icc(a, a) == pytest.approx(1.0, abs=1e-6)


def test_rmse_mae_known_values():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert calc_rmse(a, b) == pytest.approx(np.sqrt((9 + 16 + 0) / 3))
    assert calc_mae(a, b) == pytest.approx((3 + 4 + 0) / 3)


def test_nan_aware():
    a = np.array([1.0, np.nan, 3.0])
    b = np.array([1.0, 5.0, 3.0])
    assert calc_rmse(a, b) == pytest.approx(0.0, abs=1e-12)


def test_icc_constant_input_is_nan():
    a = np.ones(50)
    b = np.linspace(0, 1, 50)
    assert np.isnan(calc_icc(a, b))


def test_icc_high_for_strongly_agreeing_raters():
    rng = np.random.default_rng(0)
    a = rng.normal(size=80)
    b = a + rng.normal(scale=0.05, size=80)     # near-identical raters
    assert calc_icc(a, b) > 0.9
