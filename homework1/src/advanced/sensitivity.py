"""advanced/sensitivity.py —— Cinelli & Hazlett (2020) OVB 敏感性分析。

核心恒等式（本模块自行推导，并在已知混杂 X3 上精确验证）：
设遗漏模型（不含 U）的 t 统计量为 t0（k0 个参数，自由度 n-k0），
完整模型（含 U）共 k = k0+1 个参数，U 的两个偏 R²：
    r² = partial R²(U ~ X | Z)      处理变量侧
    s² = partial R²(U ~ Y | X, Z)   结果变量侧
则加入 U 后的 t 统计量（取"偏差把估计拉向 0"的最坏情形方向）：
    t1² = (n-k) · (1-r²)/(1-s²) · max(0, |t0|/√(n-k0) − s·r/√(1-r²))²
稳健值 RV：使 |t1| 降到临界值 t_crit 的最小 (r², s²) = (RV, RV)。
点估计归零（t1=0）有闭式：q = f(√(f²+4) − f)/2, f = |t0|/√(n-k0), RV = √q。

参考: Cinelli & Hazlett (2020), "Making sense of sensitivity: extending
omitted variable bias", JRSS-B 82(1):39-67.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import R_SENSITIVITY, Config, ensure_dirs  # noqa: E402
from ols import ols_fit  # noqa: E402


def partial_r2(
    v: np.ndarray, full: np.ndarray, reduced: np.ndarray
) -> float:
    """偏 R²：控制 reduced 中变量后，加入 full 中变量对解释 v 的增量占比。

    pR2 = (R²_full − R²_reduced) / (1 − R²_reduced)
    """
    r2f = ols_fit(v, full, add_intercept=True).r_squared
    r2r = ols_fit(v, reduced, add_intercept=True).r_squared
    return float((r2f - r2r) / (1.0 - r2r))


def t_adjusted(
    t0: float, n: int, k_full: int, r2_x: float, r2_y: float
) -> float:
    """由遗漏模型 t0 与假想混杂的 (r², s²) 预测加入 U 后的 |t|（最坏方向）。

    t0: 遗漏模型的（带符号）t 统计量；k_full: 完整模型参数个数。
    """
    k0 = k_full - 1
    r2_x = float(np.clip(r2_x, 0.0, 1.0 - 1e-12))
    r2_y = float(np.clip(r2_y, 0.0, 1.0 - 1e-12))
    f = abs(t0) / np.sqrt(n - k0)
    strength = np.sqrt(r2_x * r2_y) / np.sqrt(1.0 - r2_x)  # s·r/√(1−r²)
    bracket = f - strength
    if bracket <= 0:
        return 0.0  # 偏差已把点估计拉过 0（t 改变符号，显著性丧失）
    return float(np.sqrt((n - k_full) * (1.0 - r2_x) / (1.0 - r2_y)) * bracket)


def robustness_value(
    t0: float, n: int, k_full: int, t_crit: float | None = None
) -> dict:
    """稳健值 RV：等强度混杂 (r²=s²=RV) 使 |t1| 降到 t_crit 的最小值。

    另返回点估计归零的闭式解 RV_point。
    """
    k0 = k_full - 1
    f = abs(t0) / np.sqrt(n - k0)
    # 点估计归零： r²/√(1−r²) = f  =>  q² + f²q − f² = 0, q = r²
    q = f * (np.sqrt(f**2 + 4.0) - f) / 2.0
    rv_point = float(np.sqrt(q))

    if t_crit is None:
        return {"rv_point": rv_point, "rv_significance": None}

    def g(rv: float) -> float:
        return t_adjusted(t0, n, k_full, rv, rv) - t_crit

    # g(0) = |t0| − t_crit > 0（原模型显著才有意义）；随 rv 增大 g 单调降后升，
    # 但在跨越 0 之前先减，brentq 找第一个根
    if g(0.0) <= 0:
        return {"rv_point": rv_point, "rv_significance": 0.0}
    lo, hi = 0.0, 0.999999
    # 找第一个使 g 变号的区间
    grid = np.linspace(0, hi, 2000)
    gv = np.array([g(x) for x in grid])
    idx = np.where((gv[:-1] > 0) & (gv[1:] <= 0))[0]
    if len(idx) == 0:
        return {"rv_point": rv_point, "rv_significance": 1.0}  # 任何强度都推翻不了
    root = optimize.brentq(g, grid[idx[0]], grid[idx[0] + 1])
    return {"rv_point": rv_point, "rv_significance": float(root)}


def sensitivity_grid(
    t0: float, n: int, k_full: int, r2_x_grid: np.ndarray, r2_y_grid: np.ndarray
) -> np.ndarray:
    """(r², s²) 网格上的 |t1| 矩阵，供等高线图。行 = r²(X侧)，列 = s²(Y侧)。"""
    T = np.empty((len(r2_x_grid), len(r2_y_grid)))
    for i, rx in enumerate(r2_x_grid):
        for j, ry in enumerate(r2_y_grid):
            T[i, j] = t_adjusted(t0, n, k_full, rx, ry)
    return T


def validate_with_known_confounder(cfg: Config, df: pd.DataFrame) -> dict:
    """删去 X3 视为"不可观测 U"，检验恒等式能否精确找回完整模型 t 值。

    t0     = Y ~ X1 + X2          的 X1 t 值（遗漏模型）
    t_full = Y ~ X1 + X2 + X3     的 X1 t 值（真实完整模型）
    r²     = pR2(X3 ~ X1 | X2)    处理侧
    s²     = pR2(X3 ~ Y | X1,X2)  结果侧
    预测 t1 与 t_full 之差应 < 1e-8（同一数据上的代数恒等式）。
    """
    y = df["Y"].to_numpy()
    X1 = df[["X1"]].to_numpy()
    X2 = df[["X2"]].to_numpy()
    X3 = df[["X3"]].to_numpy()
    n = len(y)

    # 遗漏模型 t0
    res0 = ols_fit(y, np.column_stack([X1, X2]), add_intercept=True)
    se0 = np.sqrt(res0.rss / res0.df_resid * np.diag(res0.xtx_inv))
    t0 = float(res0.beta[1] / se0[1])

    # 完整模型 t_full
    resF = ols_fit(y, np.column_stack([X1, X2, X3]), add_intercept=True)
    seF = np.sqrt(resF.rss / resF.df_resid * np.diag(resF.xtx_inv))
    t_full = float(resF.beta[1] / seF[1])

    # U=X3 的两个偏 R²
    r2_x = partial_r2(X3.ravel(), np.column_stack([X1, X2]), X2)               # X3 ~ X1 | X2
    r2_y = partial_r2(X3.ravel(), np.column_stack([X1, X2, y]), np.column_stack([X1, X2]))  # X3 ~ Y | X1,X2

    t_pred = t_adjusted(t0, n, k_full=4, r2_x=r2_x, r2_y=r2_y)
    diff = abs(t_pred - abs(t_full))
    return {
        "t0_omitting_x3": t0,
        "t_full_actual": t_full,
        "r2_x3_x1_given_x2": r2_x,
        "r2_x3_y_given_x1x2": r2_y,
        "t1_predicted": t_pred,
        "abs_diff": diff,
        "exact": bool(diff < 1e-6),
    }


def run_sensitivity(cfg: Config, df: pd.DataFrame) -> dict:
    """主入口：以多元回归 (Y ~ X1+X2+X3) 中 X1 的 t 值为对象，
    评估"若存在未观测混杂 U"的敏感性与稳健值，并做已知混杂验证。"""
    ensure_dirs()
    y = df["Y"].to_numpy()
    X = df[["X1", "X2", "X3"]].to_numpy()
    n = len(y)

    res = ols_fit(y, X, add_intercept=True)
    se = np.sqrt(res.rss / res.df_resid * np.diag(res.xtx_inv))
    t0 = float(res.beta[1] / se[1])
    beta1 = float(res.beta[1])
    t_crit = float(stats.t.ppf(0.975, res.df_resid))

    rv = robustness_value(t0, n, k_full=cfg.k, t_crit=t_crit)

    # 等高线网格
    grid = np.linspace(0.0, 0.95, 60)
    T = sensitivity_grid(t0, n, cfg.k, grid, grid)
    np.save(R_SENSITIVITY / "sensitivity_grid.npy", T)
    np.save(R_SENSITIVITY / "sensitivity_grid_axis.npy", grid)

    validation = validate_with_known_confounder(cfg, df)

    payload = {
        "beta1": beta1,
        "t0": t0,
        "t_crit": t_crit,
        "rv_point": rv["rv_point"],
        "rv_significance": rv["rv_significance"],
        "validation_drop_x3": validation,
    }
    out = R_SENSITIVITY / "sensitivity.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[CH2020] beta1 = {beta1:.4f}, t0 = {t0:.4f}, t_crit = {t_crit:.4f}")
    print(f"  稳健值 RV(显著性) = {rv['rv_significance']:.4f}" if rv["rv_significance"] is not None else "  显著性不可推翻")
    print(f"  稳健值 RV(点估计) = {rv['rv_point']:.4f}")
    v = validation
    print(f"  已知混杂验证: 删 X3 后 t0={v['t0_omitting_x3']:.4f}, "
          f"由偏R²找回 t1={v['t1_predicted']:.4f} vs 实际 {abs(v['t_full_actual']):.4f}, "
          f"|Δ|={v['abs_diff']:.2e} {'✓' if v['exact'] else '✗'}")
    return payload


if __name__ == "__main__":
    from config import rng_for
    from datagen import load_main_dataset

    cfg = Config()
    run_sensitivity(cfg, load_main_dataset(cfg))
