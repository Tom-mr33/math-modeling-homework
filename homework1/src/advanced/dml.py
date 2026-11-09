"""advanced/dml.py —— Double/Debiased ML (Chernozhukov et al. 2018)。

部分线性模型  Y = θ·X1 + g(W) + ε,  X1 = m(W) + v,  W=(X2,X3)：
K 折交叉拟合估计烦恼项 m̂、ĝ，残差化后 θ̂ = Σw̃ỹ/Σw̃²（Neyman 正交 → √n 渐近正态）。
学习器可插拔：lasso（含多项式基展开）或 rf。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _make_learner(kind: str, poly_degree: int | None = None):
    if kind == "lasso":
        model = LassoCV(cv=5, max_iter=20000)
        if poly_degree:
            return make_pipeline(
                StandardScaler(),
                PolynomialFeatures(poly_degree, include_bias=False),
                model,
            )
        return make_pipeline(StandardScaler(), model)
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5, random_state=0, n_jobs=-1
        )
    raise ValueError(kind)


def dml_beta1(
    df: pd.DataFrame,
    learner: str = "lasso",
    poly_degree: int | None = None,
    n_folds: int = 5,
    seed: int = 0,
) -> dict:
    """交叉拟合 DML 估计部分线性参数 θ（即 β1）。

    Returns
    -------
    dict(theta, se, t, p, fold_resid)
    """
    from scipy import stats

    y = df["Y"].to_numpy()
    d = df["X1"].to_numpy()
    W = df[["X2", "X3"]].to_numpy()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_res = np.empty(len(y))
    d_res = np.empty(len(y))
    for train, test in kf.split(W):
        my = clone(_make_learner(learner, poly_degree)).fit(W[train], y[train])
        md = clone(_make_learner(learner, poly_degree)).fit(W[train], d[train])
        y_res[test] = y[test] - my.predict(W[test])
        d_res[test] = d[test] - md.predict(W[test])

    theta = float(d_res @ y_res / (d_res @ d_res))
    eps = y_res - theta * d_res
    n = len(y)
    # 渐近方差：Σw̃²(ỹ−θw̃)² / (Σw̃²)²
    var = float((d_res**2 * eps**2).sum() / (d_res @ d_res) ** 2)
    se = float(np.sqrt(var))
    t = theta / se
    p = 2 * stats.t.sf(abs(t), df=n - 4)
    return {"theta": theta, "se": se, "t": t, "p": p, "learner": learner}
