"""与估计解耦的推断层：SE（经典/HC 稳健）、t、p、CI、系数表。

输入是 OLSResult + 设计矩阵，不关心数据从哪来、系数谁算的。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ols import OLSResult


@dataclass(frozen=True)
class CoefRow:
    name: str
    beta: float
    se: float
    t: float
    p: float
    ci_lo: float
    ci_hi: float


def classical_se(res: OLSResult) -> np.ndarray:
    """经典同方差 SE：sigma_hat^2 * (X'X)^{-1} 对角元开方，sigma^2 = RSS/(n-k)。"""
    sigma2 = res.rss / res.df_resid
    return np.sqrt(sigma2 * np.diag(res.xtx_inv))


def hc1_se(y: np.ndarray, X: np.ndarray, res: OLSResult) -> np.ndarray:
    """异方差稳健 (HC1) SE：White 三明治 + 自由度修正 n/(n-k)。"""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    u2 = res.resid**2
    bread = res.xtx_inv
    meat = (X.T * u2) @ X  # X' diag(u^2) X
    cov = bread @ meat @ bread * (res.n / res.df_resid)
    return np.sqrt(np.diag(cov))


def coefficient_table(
    res: OLSResult,
    X: np.ndarray,
    y: np.ndarray | None = None,
    names: list[str] | None = None,
    robust: str | None = None,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    """生成回归系数表（beta / SE / t / p / CI）。

    y: 原始因变量（robust="hc1" 时需要）；names: 系数名；robust: None 经典 / "hc1" 稳健。
    """
    if names is None:
        names = [f"x{j}" for j in range(res.k)]
    if robust == "hc1":
        if y is None:
            raise ValueError("robust='hc1' 需要传入原始 y")
        se = hc1_se(y, X, res)
    else:
        se = classical_se(res)

    crit = stats.t.ppf(0.5 + ci_level / 2, res.df_resid)
    rows = []
    for j, name in enumerate(names):
        t_stat = res.beta[j] / se[j]
        p = 2.0 * stats.t.sf(abs(t_stat), res.df_resid)
        rows.append(
            CoefRow(
                name=name,
                beta=float(res.beta[j]),
                se=float(se[j]),
                t=float(t_stat),
                p=float(p),
                ci_lo=float(res.beta[j] - crit * se[j]),
                ci_hi=float(res.beta[j] + crit * se[j]),
            )
        )
    df = pd.DataFrame([r.__dict__ for r in rows])
    df.attrs["r2"] = res.r_squared
    df.attrs["adj_r2"] = res.adj_r_squared
    df.attrs["n"] = res.n
    df.attrs["df_resid"] = res.df_resid
    return df


def simple_regression_summary(y: np.ndarray, x: np.ndarray, ci_level: float = 0.95) -> pd.DataFrame:
    """一元线性回归 Y ~ x（带截距）的系数表，供简单回归分析使用。"""
    from ols import ols_fit  # 延迟导入，保持推断层可独立测试

    res = ols_fit(y, x, add_intercept=True)
    Xd = np.column_stack([np.ones(len(y)), np.asarray(x, dtype=float)])
    return coefficient_table(
        res, Xd, y=y, names=["const", "x"], ci_level=ci_level
    )
