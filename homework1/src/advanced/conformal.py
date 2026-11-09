"""advanced/conformal.py —— 分裂 conformal 反事实预测区间（Lei & Candès 2021）。

方法：每次重复中
  1. 生成训练集(n=60)与独立测试集(n=60)（同一数据生成机制）；
  2. 训练集再分为拟合折(30)/校准折(30)，拟合线性模型并计算校准残差；
  3. 区间半径 q = 分位数(⌈(n_cal+1)(1−α)⌉/n_cal · |校准残差|)；
  4. 测试集覆盖率统计 → 有限样本边际覆盖保证的经验验证。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import R_CONFORMAL, Config, ensure_dirs, rng_for  # noqa: E402
from datagen import generate_dataset  # noqa: E402
from ols import ols_fit  # noqa: E402


def split_conformal_interval(
    y: np.ndarray,
    X: np.ndarray,
    X_new: np.ndarray,
    alpha: float = 0.10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """分裂 conformal：返回 (下界, 上界, 半径 q)。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_fit = len(y) // 2
    fit_idx, cal_idx = idx[:n_fit], idx[n_fit:]

    res = ols_fit(y[fit_idx], X[fit_idx], add_intercept=True)
    beta = res.beta
    # 校准残差：模型在拟合折上估计，残差必须在未参与拟合的校准折上计算
    cal_resid = np.abs(y[cal_idx] - np.column_stack([np.ones(len(cal_idx)), X[cal_idx]]) @ beta)

    n_cal = len(cal_idx)
    q = float(
        np.quantile(cal_resid, np.ceil((n_cal + 1) * (1 - alpha)) / n_cal, method="higher")
    )
    new_pred = np.column_stack([np.ones(len(X_new)), X_new]) @ beta
    return new_pred - q, new_pred + q, q


def run_conformal(cfg: Config, alpha: float = 0.10, reps: int = 200) -> dict:
    """Monte Carlo 验证 conformal 区间的经验覆盖率 ≥ 1−α。"""
    ensure_dirs()
    rng = rng_for(cfg, "conformal")
    seeds = rng.integers(0, 2**31, size=reps)

    coverages, widths = [], []
    example = None
    for b, s in enumerate(seeds):
        r = np.random.default_rng(int(s))
        train = generate_dataset(cfg, r)
        test = generate_dataset(cfg, r)
        cols = ["X1", "X2", "X3"]
        lo, hi, q = split_conformal_interval(
            train["Y"].to_numpy(),
            train[cols].to_numpy(),
            test[cols].to_numpy(),
            alpha=alpha,
            seed=int(s) + 1,
        )
        truth = test["Y"].to_numpy()
        cov = float(((truth >= lo) & (truth <= hi)).mean())
        coverages.append(cov)
        widths.append(float((hi - lo).mean()))
        if b == 0:  # 保留一次实验用于绘图
            order = np.argsort(test["X1"].to_numpy())
            example = {
                "x1": test["X1"].to_numpy()[order].tolist(),
                "truth": truth[order].tolist(),
                "lo": lo[order].tolist(),
                "hi": hi[order].tolist(),
            }

    coverages_arr = np.array(coverages)
    payload = {
        "alpha": alpha,
        "nominal": 1 - alpha,
        "reps": reps,
        "mean_coverage": float(coverages_arr.mean()),
        "sd_coverage": float(coverages_arr.std(ddof=1)),
        "min_coverage": float(coverages_arr.min()),
        "mean_width": float(np.mean(widths)),
        "coverage_ok": bool(coverages_arr.mean() >= 1 - alpha - 0.02),
        "example": example,
    }
    out = R_CONFORMAL / "conformal.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[Conformal] 名义 {1-alpha:.0%}, 经验覆盖 {payload['mean_coverage']:.3f}"
        f" ± {payload['sd_coverage']:.3f}, 平均区间宽度 {payload['mean_width']:.2f}"
    )
    return payload


if __name__ == "__main__":
    run_conformal(Config())
