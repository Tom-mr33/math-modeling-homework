"""CH2020 敏感性分析模块的互验测试。

1. 精确恒等式：删去 X3 视为未观测混杂，t_adjusted 必须精确找回完整模型 t 值
2. RV 闭式与数值解一致：等强度混杂使 t 恰为 t_crit 时，rv/√(1−rv) = f − t_crit/√(n−k)
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import Config, rng_for  # noqa: E402
from datagen import generate_dataset  # noqa: E402
from ols import ols_fit  # noqa: E402
from advanced.sensitivity import partial_r2, robustness_value, t_adjusted  # noqa: E402


@pytest.fixture(scope="module")
def df():
    cfg = Config()
    return generate_dataset(cfg, rng_for(cfg, "datagen"))


def test_identity_recovers_t_after_adding_confounder(df):
    y = df["Y"].to_numpy()
    X1, X2, X3 = (df[[c]].to_numpy() for c in ("X1", "X2", "X3"))
    n = len(y)

    def t_of(cols):
        res = ols_fit(y, cols, add_intercept=True)
        se = np.sqrt(res.rss / res.df_resid * np.diag(res.xtx_inv))
        return float(res.beta[1] / se[1])

    t0 = t_of(np.column_stack([X1, X2]))
    t_full = abs(t_of(np.column_stack([X1, X2, X3])))

    r2_x = partial_r2(X3.ravel(), np.column_stack([X1, X2]), X2)
    r2_y = partial_r2(X3.ravel(), np.column_stack([X1, X2, y]), np.column_stack([X1, X2]))
    t_pred = t_adjusted(t0, n, k_full=4, r2_x=r2_x, r2_y=r2_y)

    assert abs(t_pred - t_full) < 1e-6


def test_rv_matches_closed_form(df):
    """RV 数值解应满足 rv/√(1−rv) = |t0|/√(n−k0) − t_crit/√(n−k)。"""
    from scipy import stats

    y = df["Y"].to_numpy()
    X = df[["X1", "X2", "X3"]].to_numpy()
    n, k = len(y), 4
    res = ols_fit(y, X, add_intercept=True)
    se = np.sqrt(res.rss / res.df_resid * np.diag(res.xtx_inv))
    t0 = float(res.beta[1] / se[1])
    t_crit = float(stats.t.ppf(0.975, res.df_resid))

    rv = robustness_value(t0, n, k_full=k, t_crit=t_crit)["rv_significance"]
    lhs = rv / np.sqrt(1 - rv)
    rhs = abs(t0) / np.sqrt(n - (k - 1)) - t_crit / np.sqrt(n - k)
    assert abs(lhs - rhs) < 1e-4
