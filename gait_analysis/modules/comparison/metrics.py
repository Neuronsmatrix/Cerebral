"""Accuracy metrics for Vicon comparison: RMSE/MAE/Pearson (NaN-aware) + ICC."""
import numpy as np

_ICC_TYPE = {"1,1": "ICC(1,1)", "2,1": "ICC(A,1)", "3,1": "ICC(C,1)"}


def _paired(a, b):
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    m = ~(np.isnan(a) | np.isnan(b))
    return a[m], b[m]


def calc_rmse(a, b) -> float:
    a, b = _paired(a, b)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def calc_mae(a, b) -> float:
    a, b = _paired(a, b)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def calc_pearson(a, b) -> float:
    a, b = _paired(a, b)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def calc_icc(a, b, icc_type: str = "3,1") -> float:
    """ICC between two raters a, b (1-D, paired). Uses pingouin; NaN if degenerate."""
    import pandas as pd
    import pingouin as pg

    a, b = _paired(a, b)
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    n = a.size
    long = pd.DataFrame({
        "target": list(range(n)) * 2,
        "rater": ["a"] * n + ["b"] * n,
        "rating": np.concatenate([a, b]),
    })
    res = pg.intraclass_corr(data=long, targets="target", raters="rater",
                             ratings="rating").set_index("Type")
    return float(res.loc[_ICC_TYPE[icc_type], "ICC"])
