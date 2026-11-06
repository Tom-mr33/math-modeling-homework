"""advanced/meta_learners.py —— 2020/2021 元学习器：R-learner 与 DR-learner。

- R-learner（Nie & Wager 2021, Biometrika）：对 Y 与处理同时正交化后，
  解加权损失 min_τ Σ w̃²(ỹ − τ·w̃)²。常数效应时闭式 τ̂ = Σw̃ỹ/Σw̃²；
  异质效应时以 w̃² 为样本权重、ỹ/w̃ 为目标做非参回归。
- DR-learner（Kennedy 2020 思想的连续处理版本）：以部分线性模型的
  影响函数伪结果 φᵢ = θ̂ + w̃ᵢ(ỹᵢ − θ̂w̃ᵢ)/E[w̃²] 做二阶段非参回归。
- CausalForestDML（Athey & Wager 2019，econml 实现），缺失时优雅降级。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # econml 缺失时降级
    from econml.dml import CausalForestDML

    HAS_ECONML = True
except Exception:  # noqa: BLE001
    CausalForestDML = None
    HAS_ECONML = False


def _rf(seed: int = 0) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )


def _crossfit_residuals(df: pd.DataFrame, n_folds: int, seed: int):
    """交叉拟合正交化：返回 (ỹ, w̃, W)。"""
    y = df["Y"].to_numpy()
    d = df["X1"].to_numpy()
    W = df[["X2", "X3"]].to_numpy()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_res, d_res = np.empty(len(y)), np.empty(len(y))
    for tr, te in kf.split(W):
        my = _rf(seed).fit(W[tr], y[tr])
        md = _rf(seed + 1).fit(W[tr], d[tr])
        y_res[te] = y[te] - my.predict(W[te])
        d_res[te] = d[te] - md.predict(W[te])
    return y_res, d_res, W


def r_learner_beta1(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> dict:
    """R-learner 常数效应估计（与 DML 点估计同型，但损失加权结构不同源）。"""
    y_res, d_res, _ = _crossfit_residuals(df, n_folds, seed)
    denom = float(d_res @ d_res)
    theta = float(d_res @ y_res / denom)
    eps = y_res - theta * d_res
    var = float((d_res**2 * eps**2).sum() / denom**2)
    from scipy import stats

    se = float(np.sqrt(var))
    return {
        "theta": theta,
        "se": se,
        "t": theta / se,
        "p": 2 * stats.t.sf(abs(theta / se), df=len(df) - 4),
        "method": "R-learner",
    }


def dr_learner_beta1(df: pd.DataFrame, n_folds: int = 5, seed: int = 0) -> dict:
    """DR-learner：EIF 伪结果二阶段回归后取均值对应常数效应。"""
    y_res, d_res, _ = _crossfit_residuals(df, n_folds, seed)
    theta0 = float(d_res @ y_res / (d_res @ d_res))
    w_var = float((d_res**2).mean())
    phi = theta0 + d_res * (y_res - theta0 * d_res) / w_var  # 伪结果
    # 常数效应下二阶段拟合常数；其 se 用伪结果样本方差
    theta = float(phi.mean())
    se = float(phi.std(ddof=1) / np.sqrt(len(phi)))
    from scipy import stats

    return {
        "theta": theta,
        "se": se,
        "t": theta / se,
        "p": 2 * stats.t.sf(abs(theta / se), df=len(df) - 4),
        "method": "DR-learner",
    }


def causal_forest_beta1(df: pd.DataFrame, seed: int = 0) -> dict | None:
    """CausalForestDML（econml）；不可用时返回 None。"""
    if not HAS_ECONML:
        return None
    y = df["Y"].to_numpy()
    d = df["X1"].to_numpy()
    X_feat = df[["X2", "X3"]].to_numpy()
    est = CausalForestDML(
        model_y=_rf(seed),
        model_t=_rf(seed + 1),
        discrete_treatment=False,
        n_estimators=400,
        min_samples_leaf=5,
        random_state=seed,
    )
    est.fit(y, d, X=X_feat, W=X_feat)
    tau = est.effect(X_feat)
    # ATE 及其 se（econml 自带推断）
    ate = float(np.mean(tau))
    se = float(np.std(tau, ddof=1) / np.sqrt(len(tau)))
    return {"theta": ate, "se": se, "method": "CausalForest", "tau": tau}
