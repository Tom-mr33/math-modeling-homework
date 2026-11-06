"""自实现 OLS / FWL / 偏相关 vs statsmodels 数值互验。

运行：在 作业1/ 目录下  python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest

from config import Config, rng_for
from datagen import generate_dataset
from inference import classical_se, coefficient_table, simple_regression_summary
from ols import (
    fwl_beta1,
    ols_fit,
    partial_correlation,
    residualize,
    t_from_partial_r2,
)

RTOL = 1e-10


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    cfg = Config()
    return generate_dataset(cfg, rng_for(cfg, "datagen"))


def test_rng_deterministic(df: pd.DataFrame):
    """同种子两次生成必须完全一致（可复现性）。"""
    cfg = Config()
    df2 = generate_dataset(cfg, rng_for(cfg, "datagen"))
    pd.testing.assert_frame_equal(df, df2)


def test_ols_vs_statsmodels(df: pd.DataFrame):
    """多元 OLS 系数、SE、t、p、R² 与 statsmodels 互验。"""
    y = df["Y"].to_numpy()
    X = df[["X1", "X2", "X3"]].to_numpy()

    res = ols_fit(y, X, add_intercept=True)
    sm_res = sm.OLS(y, sm.add_constant(X)).fit()

    np.testing.assert_allclose(res.beta, sm_res.params, rtol=RTOL)
    np.testing.assert_allclose(classical_se(res), sm_res.bse, rtol=1e-8)
    np.testing.assert_allclose(res.rss, sm_res.ssr, rtol=RTOL)
    assert abs(res.r_squared - sm_res.rsquared) < 1e-10
    assert abs(res.adj_r_squared - sm_res.rsquared_adj) < 1e-10

    table = coefficient_table(
        res,
        np.column_stack([np.ones(res.n), X]),
        y=y,
        names=["const", "X1", "X2", "X3"],
    )
    np.testing.assert_allclose(table["t"].to_numpy(), sm_res.tvalues, rtol=1e-8)
    np.testing.assert_allclose(table["p"].to_numpy(), sm_res.pvalues, rtol=1e-8)


def test_simple_regression_vs_statsmodels(df: pd.DataFrame):
    """一元回归摘要与 statsmodels 互验（步骤 2 基准）。"""
    sm_res = sm.OLS(df["Y"], sm.add_constant(df["X1"])).fit()
    table = simple_regression_summary(df["Y"].to_numpy(), df["X1"].to_numpy())
    np.testing.assert_allclose(table["beta"].to_numpy(), sm_res.params, rtol=RTOL)
    np.testing.assert_allclose(table["t"].to_numpy(), sm_res.tvalues, rtol=1e-8)


def test_fwl_equals_multiple_ols(df: pd.DataFrame):
    """FWL 两步估计 == 多元回归 beta1（作业验收标准：|Δ| < 1e-8）。"""
    y = df["Y"].to_numpy()
    x1 = df["X1"].to_numpy()
    Z = df[["X2", "X3"]].to_numpy()

    beta_fwl, e_y, e_x1 = fwl_beta1(y, x1, Z)
    res = ols_fit(y, X=df[["X1", "X2", "X3"]].to_numpy(), add_intercept=True)
    beta_multi = res.beta[1]

    assert abs(beta_fwl - beta_multi) < 1e-8
    # 残差必须正交于控制变量与常数向量（FWL 推导的核心性质）
    Zc = np.column_stack([np.ones(len(y)), Z])
    np.testing.assert_allclose(Zc.T @ e_y, 0.0, atol=1e-8)
    np.testing.assert_allclose(Zc.T @ e_x1, 0.0, atol=1e-8)


def test_partial_correlation_and_t(df: pd.DataFrame):
    """偏相关 t 统计量 == 软件输出的 t 值（作业验收标准：|Δ| < 1e-6）。"""
    y = df["Y"].to_numpy()
    x1 = df["X1"].to_numpy()
    Z = df[["X2", "X3"]].to_numpy()
    n, k = len(y), 4

    r = partial_correlation(y, x1, Z)
    t_manual = t_from_partial_r2(r, n, k)

    res = ols_fit(y, df[["X1", "X2", "X3"]].to_numpy(), add_intercept=True)
    table = coefficient_table(
        res,
        np.column_stack([np.ones(n), df[["X1", "X2", "X3"]].to_numpy()]),
        y=y,
        names=["const", "X1", "X2", "X3"],
    )
    t_software = table.loc[table["name"] == "X1", "t"].item()

    assert abs(t_manual - t_software) < 1e-6


def test_residualize_matches_manual_formula(df: pd.DataFrame):
    """残差化算子与手算公式 M_Z u = u - Z(Z'Z)^{-1}Z'u（含截距）一致。"""
    u = df["Y"].to_numpy()
    Z = df[["X2", "X3"]].to_numpy()
    Zc = sm.add_constant(Z)
    u_manual = u - Zc @ np.linalg.solve(Zc.T @ Zc, Zc.T @ u)
    np.testing.assert_allclose(residualize(u, Z), u_manual, rtol=1e-6, atol=1e-8)


def test_beta1_close_to_truth(df: pd.DataFrame):
    """多元回归 beta1 应接近真值 2（正态误差下预期在几个 SE 内）。"""
    res = ols_fit(df["Y"].to_numpy(), df[["X1", "X2", "X3"]].to_numpy(), add_intercept=True)
    se = classical_se(res)
    assert abs(res.beta[1] - 2.0) < 4 * se[1]
