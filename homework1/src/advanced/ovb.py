"""遗漏变量偏误（OVB）解析公式 + Monte Carlo 数值验证。

理论（方案设计文档 2.1 节）：简单回归同时遗漏 X2 与 X3，故
    plim beta1_simple = beta1 + beta2*Cov(X1,X2)/Var(X1) + beta3*Cov(X1,X3)/Var(X1)
本题总体参数代入：beta1=2, beta2=1.5, beta3=-1, Cov(X1,X2)=-0.5, Cov(X1,X3)=-0.2, Var(X1)=1
    bias = 1.5*(-0.5) + (-1)*(-0.2) = -0.55   =>   plim beta1_simple = 1.45
（X2 项 -0.75 为主，X3 项 +0.2 部分抵消。）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import R_MC, Config, ensure_dirs, rng_for  # noqa: E402
from datagen import generate_dataset  # noqa: E402
from simulation import beta1_multiple, beta1_simple, run_mc  # noqa: E402


def ovb_asymptotic(cfg: Config) -> float:
    """总体层面的 OVB 渐近预测（含全部被遗漏的控制变量）：

    plim beta1_simple = beta1 + beta2*Cov(X1,X2)/Var(X1) + beta3*Cov(X1,X3)/Var(X1)
    总体协方差矩阵顺序为 (X2, X3, X1)。
    """
    cov = np.asarray(cfg.cov)
    var_x1 = cov[2, 2]
    b1, b2, b3 = cfg.beta_true
    cov_x1x2 = cov[0, 2]  # Cov(X2, X1)
    cov_x1x3 = cov[1, 2]  # Cov(X3, X1)
    return b1 + b2 * cov_x1x2 / var_x1 + b3 * cov_x1x3 / var_x1


def verify_ovb(cfg: Config | None = None, reps: int | None = None) -> dict:
    """MC 均值 vs 渐近理论 1.25 互验。"""
    cfg = cfg or Config()
    reps = reps or cfg.mc_reps
    ensure_dirs()

    res_simple = run_mc(cfg, beta1_simple, reps=reps)
    res_multi = run_mc(cfg, beta1_multiple, reps=reps)
    theory = ovb_asymptotic(cfg)

    payload = {
        "reps": reps,
        "theory_asymptotic_beta1_simple": theory,
        "mc_mean_simple": res_simple["mean"],
        "mc_std_simple": res_simple["std"],
        "mc_mean_multiple": res_multi["mean"],
        "mc_std_multiple": res_multi["std"],
        "truth_beta1": cfg.beta_true[0],
        "se_of_mc_mean_simple": res_simple["std"] / np.sqrt(reps),
        "se_of_mc_mean_multiple": res_multi["std"] / np.sqrt(reps),
    }

    # 判定：MC 均值与渐近理论差 < 3 个 MC 标准误（≈正态 3σ）
    z = abs(res_simple["mean"] - theory) / payload["se_of_mc_mean_simple"]
    payload["z_simple_vs_theory"] = float(z)
    payload["consistent_with_theory"] = bool(z < 3.0)

    # 抽样分布样本落盘（供直方图绘图）
    np.save(
        R_MC / "mc_samples.npy",
        np.column_stack([res_simple["samples"], res_multi["samples"]]),
    )
    out = R_MC / "ovb_mc.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OVB] 渐近理论 beta1_simple -> {theory:.4f}")
    print(f"      MC 均值  simple   = {res_simple['mean']:.4f} ± {payload['se_of_mc_mean_simple']:.4f} (MC-SE)")
    print(f"      MC 均值  multiple = {res_multi['mean']:.4f} ± {payload['se_of_mc_mean_multiple']:.4f} (MC-SE)")
    print(f"      simple 与理论 z = {z:.2f} -> {'一致' if z < 3 else '偏离！'}")
    print(f"      multiple 围绕真值 {cfg.beta_true[0]}：|均值-真值| = {abs(res_multi['mean']-cfg.beta_true[0]):.4f}")
    return payload


if __name__ == "__main__":
    verify_ovb()
