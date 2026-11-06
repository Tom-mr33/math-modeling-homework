"""main.py —— 作业步骤 1–6 流水线。

运行（在 作业1/ 目录下）：  python src/main.py
产物：
  data/sleep_score.csv            数据文件
  results/step*.json|csv          报告所需的全部数值
所有数值由种子 2024 唯一确定，可复现。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import RESULTS_DIR, Config, ensure_dirs, rng_for
from datagen import load_main_dataset
from inference import coefficient_table, simple_regression_summary
from ols import fwl_beta1, ols_fit, partial_correlation, t_from_partial_r2


def save_json(obj: dict, name: str) -> None:
    path = RESULTS_DIR / name
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"  -> {path.relative_to(RESULTS_DIR.parent)}")


def step1_describe(cfg: Config, df: pd.DataFrame) -> None:
    """步骤一：数据生成 + 描述统计 + 相关矩阵。"""
    print("[步骤1] 数据生成与描述统计")
    desc = df.describe().loc[["mean", "std"]].round(4)
    corr = df.corr().round(4)
    print(desc.to_string())
    print("相关矩阵：")
    print(corr.to_string())
    payload = {
        "n": cfg.n,
        "seed": cfg.seed,
        "describe": {c: desc[c].to_dict() for c in df.columns},
        "corr_matrix": corr.to_dict(),
        "r_12": corr.loc["X1", "X2"],  # 睡眠-自习
        "r_13": corr.loc["X1", "X3"],  # 睡眠-手机
    }
    save_json(payload, "step1_describe.json")
    corr.to_csv(RESULTS_DIR / "step1_corr.csv", encoding="utf-8")


def step2_simple(cfg: Config, df: pd.DataFrame) -> None:
    """步骤二：简单回归 Y ~ X1。"""
    print("[步骤2] 简单回归（不控制）")
    table = simple_regression_summary(
        df["Y"].to_numpy(), df["X1"].to_numpy(), ci_level=cfg.ci_level
    )
    print(table.round(4).to_string(index=False))
    beta1 = table.loc[table["name"] == "x", "beta"].item()
    print(f"  beta1_simple = {beta1:.4f}  (真值 2，预期被 X2 拉低)")
    table.to_csv(RESULTS_DIR / "step2_simple.csv", index=False, encoding="utf-8")
    save_json(
        {"beta1_simple": beta1, "r2": float(table.attrs["r2"])}, "step2_simple.json"
    )


def step3_multiple(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    """步骤三：多元回归 Y ~ X1 + X2 + X3。"""
    print("[步骤3] 多元回归（控制）")
    X = df[["X1", "X2", "X3"]].to_numpy()
    y = df["Y"].to_numpy()
    res = ols_fit(y, X, add_intercept=True)
    table = coefficient_table(
        res,
        np.column_stack([np.ones(cfg.n), X]),
        y=y,
        names=["const", "X1", "X2", "X3"],
        ci_level=cfg.ci_level,
    )
    print(table.round(4).to_string(index=False))
    print(f"  adj R2 = {res.adj_r_squared:.4f}")
    table.to_csv(RESULTS_DIR / "step3_multiple.csv", index=False, encoding="utf-8")
    beta1 = table.loc[table["name"] == "X1"].iloc[0]
    save_json(
        {
            "beta1_multiple": float(beta1["beta"]),
            "t": float(beta1["t"]),
            "p": float(beta1["p"]),
            "adj_r2": float(res.adj_r_squared),
        },
        "step3_multiple.json",
    )
    return table


def step4_fwl(cfg: Config, df: pd.DataFrame) -> float:
    """步骤四：FWL 残差化验证（|Δ| < 1e-8）。"""
    print("[步骤4] FWL 残差化验证")
    y = df["Y"].to_numpy()
    x1 = df["X1"].to_numpy()
    Z = df[["X2", "X3"]].to_numpy()
    beta_fwl, e_y, e_x1 = fwl_beta1(y, x1, Z)
    res_multi = ols_fit(y, df[["X1", "X2", "X3"]].to_numpy(), add_intercept=True)
    beta_multi = float(res_multi.beta[1])
    diff = abs(beta_fwl - beta_multi)
    print(f"  beta1_FWL      = {beta_fwl:.12f}")
    print(f"  beta1_multiple = {beta_multi:.12f}")
    print(f"  |difference|   = {diff:.2e}  ({'通过' if diff < 1e-8 else '未通过'} 1e-8)")
    assert diff < 1e-8, f"FWL 与多元回归差异 {diff} 超出 1e-8"
    save_json(
        {
            "beta1_fwl": beta_fwl,
            "beta1_multiple": beta_multi,
            "abs_diff": diff,
            "residuals_y_mean": float(e_y.mean()),
            "residuals_x1_mean": float(e_x1.mean()),
        },
        "step4_fwl.json",
    )
    np.save(RESULTS_DIR / "step4_residuals.npy", np.column_stack([e_x1, e_y]))
    return beta_fwl


def step5_partial_t(cfg: Config, df: pd.DataFrame) -> float:
    """步骤五：偏相关与 t 检验等价性（|Δ| < 1e-6）。"""
    print("[步骤5] 偏相关与 t 检验等价性")
    y = df["Y"].to_numpy()
    x1 = df["X1"].to_numpy()
    Z = df[["X2", "X3"]].to_numpy()

    r = partial_correlation(y, x1, Z)
    t_manual = t_from_partial_r2(r, cfg.n, cfg.k)

    X = df[["X1", "X2", "X3"]].to_numpy()
    res = ols_fit(y, X, add_intercept=True)
    table = coefficient_table(
        res,
        np.column_stack([np.ones(cfg.n), X]),
        y=y,
        names=["const", "X1", "X2", "X3"],
    )
    t_software = float(table.loc[table["name"] == "X1", "t"].item())
    diff = abs(t_manual - t_software)
    print(f"  r_1y.23   = {r:.6f}")
    print(f"  t_manual  = {t_manual:.6f}")
    print(f"  t_software= {t_software:.6f}")
    print(f"  |difference| = {diff:.2e}  ({'通过' if diff < 1e-6 else '未通过'} 1e-6)")
    assert diff < 1e-6, f"手算 t 与软件 t 差异 {diff} 超出 1e-6"
    save_json(
        {
            "r_partial": r,
            "t_manual": t_manual,
            "t_software": t_software,
            "abs_diff": diff,
        },
        "step5_partial_t.json",
    )
    return r


def step6_compare(cfg: Config, df: pd.DataFrame, step3_table: pd.DataFrame) -> None:
    """步骤六：简单 vs 多元系数对比。"""
    print("[步骤6] 系数对比讨论")
    beta_simple = pd.read_csv(RESULTS_DIR / "step2_simple.csv")
    beta1_simple = float(beta_simple.loc[beta_simple["name"] == "x", "beta"].item())
    beta1_multi = float(step3_table.loc[step3_table["name"] == "X1", "beta"].item())
    se_simple = float(beta_simple.loc[beta_simple["name"] == "x", "se"].item())
    se_multi = float(step3_table.loc[step3_table["name"] == "X1", "se"].item())

    cov12 = df["X1"].cov(df["X2"])
    cov13 = df["X1"].cov(df["X3"])
    var1 = df["X1"].var(ddof=1)
    # OVB 渐近偏差：beta2*cov12/var1 + beta3*cov13/var1（简单回归遗漏 X2、X3 两项）
    b2, b3 = cfg.beta_true[1], cfg.beta_true[2]
    bias_x2 = b2 * cov12 / var1
    bias_x3 = b3 * cov13 / var1
    from advanced.ovb import ovb_asymptotic

    plim_simple = ovb_asymptotic(cfg)

    print(f"  简单回归 beta1  = {beta1_simple:.4f} (SE {se_simple:.4f})")
    print(f"  多元回归 beta1  = {beta1_multi:.4f} (SE {se_multi:.4f})")
    print(f"  控制后变化      = {beta1_multi - beta1_simple:+.4f}")
    print(f"  样本 OVB 分解   = X2项 {bias_x2:+.4f} + X3项 {bias_x3:+.4f} = {bias_x2+bias_x3:+.4f}")
    print(f"  渐近理论 plim beta1_simple = {plim_simple:.4f}")
    save_json(
        {
            "beta1_simple": beta1_simple,
            "beta1_multiple": beta1_multi,
            "se_simple": se_simple,
            "se_multiple": se_multi,
            "change": beta1_multi - beta1_simple,
            "ovb_sample_bias_x2": bias_x2,
            "ovb_sample_bias_x3": bias_x3,
            "ovb_sample_bias_total": bias_x2 + bias_x3,
            "theory_plim_simple": plim_simple,
        },
        "step6_compare.json",
    )


def main() -> None:
    cfg = Config()
    ensure_dirs()
    print(f"配置：n={cfg.n}, seed={cfg.seed}, 真值 beta={cfg.beta_full.tolist()}\n")

    df = load_main_dataset(cfg)
    print(f"已生成/保存 data/sleep_score.csv ({len(df)} 行)\n")

    step1_describe(cfg, df)
    step2_simple(cfg, df)
    t3 = step3_multiple(cfg, df)
    step4_fwl(cfg, df)
    step5_partial_t(cfg, df)
    step6_compare(cfg, df, t3)
    print("\n全部步骤完成，数值已落盘 results/")


if __name__ == "__main__":
    main()
