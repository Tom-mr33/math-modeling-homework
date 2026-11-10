"""非线性混淆场景下的估计器对比（蒙特卡洛版）。

场景（方案设计文档扩展 3）：X2 服从右偏分布（指数变换，均值/方差不变，偏度≈2.3），
Y = 60 + 2·X1 + 1.5·X2² − X3 + ε，真实部分线性参数 θ0 = β1 = 2。

设计说明（重要）：联合正态下任何 g(W) 与 X1 的线性投影残差正交，线性 OLS
对 β1 无偏（数值验证 plim≈2.0），因此必须让 X2 右偏（现实中也更合理：
多数学生睡眠集中，少数很长）线性误设才会传入 β1。
线性 OLS 渐近参考值由 N=30 万同分布仿真确定（≈2.81）。

竞赛形式（n_race=600；n=60 时 ML 烦恼项自身方差过大，见报告讨论）：
  · MC 竞赛（主）：M=60 次重复，报告各方法 MC 均值 ± MC 标准误与偏差。
    方法：线性 OLS / DML-Lasso(线性烦恼) / DML-RF。
    注 1：常数效应下 R-learner 与 DR-learner 的点估计与 DML-RF 恒等
    （Robinson 正交化的同一闭式），故并入 DML-RF 一行，区别见单次表。
    注 2：DML-Lasso(3次基) 对重尾 X2² 不稳健（Lasso 被极端值支配），已剔除。
  · 单次展示表（race_n 数据，独立种子）：全部七法，含 CausalForest。
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advanced.dml import dml_beta1  # noqa: E402
from advanced.meta_learners import (  # noqa: E402
    causal_forest_beta1,
    dr_learner_beta1,
    r_learner_beta1,
)
from config import R_RACE, Config, ensure_dirs, rng_for  # noqa: E402
from datagen import generate_nonlinear_dataset  # noqa: E402
from ols import ols_fit  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def ols_linear_beta1(df: pd.DataFrame) -> float:
    res = ols_fit(
        df["Y"].to_numpy(), df[["X1", "X2", "X3"]].to_numpy(), add_intercept=True
    )
    return float(res.beta[1])


def _ols_plim_reference(cfg: Config, n_ref: int = 300_000, seed: int = 99) -> float:
    """大样本仿真确定线性 OLS 的渐近参考值（同一 DGP，独立种子）。"""
    class _Big:
        pass

    big = _Big()
    big.n = n_ref
    big.mu, big.cov = cfg.mu, cfg.cov
    big.sigma_eps, big.beta_true, big.beta0_true = (
        cfg.sigma_eps,
        cfg.beta_true,
        cfg.beta0_true,
    )
    rng = np.random.default_rng(seed)
    df = generate_nonlinear_dataset(big, rng, g=lambda x: x**2, skew_x2=True)
    return ols_linear_beta1(df)


def run_race(cfg: Config | None = None, mc_reps: int = 60, race_n: int = 600) -> dict:
    cfg = cfg or Config()
    ensure_dirs()
    truth = cfg.beta_true[0]
    plim_ols = _ols_plim_reference(cfg)

    class _RaceCfg:
        pass

    def _race_cfg(n: int):
        rc = _RaceCfg()
        rc.n = n
        rc.mu, rc.cov = cfg.mu, cfg.cov
        rc.sigma_eps, rc.beta_true, rc.beta0_true = (
            cfg.sigma_eps,
            cfg.beta_true,
            cfg.beta0_true,
        )
        return rc

    # ---------- 单次展示表（race_n 数据，独立种子，全部七法） ----------
    rc = _race_cfg(race_n)
    df = generate_nonlinear_dataset(
        rc, np.random.default_rng(20240601), g=lambda x: x**2, skew_x2=True
    )
    single: list[dict] = [{"method": "OLS(线性)", "theta": ols_linear_beta1(df), "se": None}]
    for name, kw in [
        ("DML-Lasso(线性烦恼)", {}),
        ("DML-RF", {"learner": "rf"}),
    ]:
        r = dml_beta1(df, **kw)
        single.append({"method": name, "theta": r["theta"], "se": r["se"]})
    r = r_learner_beta1(df)
    single.append({"method": "R-learner(RF)", "theta": r["theta"], "se": r["se"]})
    r = dr_learner_beta1(df)
    single.append({"method": "DR-learner(RF)", "theta": r["theta"], "se": r["se"]})
    cf = causal_forest_beta1(df)
    if cf is not None:
        single.append({"method": "CausalForest", "theta": cf["theta"], "se": cf["se"]})

    # ---------- MC 竞赛（M 次重复，渐近偏差对比） ----------
    base = rng_for(cfg, "nonlinear")
    child_seeds = base.integers(0, 2**31, size=mc_reps)
    methods = ["OLS(线性)", "DML-Lasso(线性烦恼)", "DML-RF"]
    samples = {m: [] for m in methods}
    for s in child_seeds:
        r = np.random.default_rng(int(s))
        d = generate_nonlinear_dataset(rc, r, g=lambda x: x**2, skew_x2=True)
        samples["OLS(线性)"].append(ols_linear_beta1(d))
        samples["DML-Lasso(线性烦恼)"].append(
            dml_beta1(d, learner="lasso")["theta"]
        )
        samples["DML-RF"].append(dml_beta1(d, learner="rf")["theta"])

    mc = {}
    for m in methods:
        arr = np.asarray(samples[m])
        mc[m] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "mc_se": float(arr.std(ddof=1) / np.sqrt(len(arr))),
            "bias": float(arr.mean() - truth),
        }

    payload = {
        "truth": truth,
        "ols_plim_reference": plim_ols,
        "n": race_n,
        "mc_reps": mc_reps,
        "mc": mc,
        "single_draw": single,
        "note": "常数效应下 R-learner/DR-learner 点估计与 DML-RF 恒等（同一 Robinson 闭式）；n=60 下各 ML 方法方差过大，竞赛在 n=600 下进行",
    }
    out = R_RACE / "race.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Race] 右偏混淆 Y = 60 + 2·X1 + 1.5·X2² − X3 + ε (n={race_n})")
    print(f"  线性 OLS 渐近参考值（N=30万）: {plim_ols:.4f}")
    print(f"  MC 竞赛（M={mc_reps}）: {'方法':<22}{'MC均值':>10}{'MC-SE':>9}{'偏差':>10}")
    for m in methods:
        print(
            f"  {m:<22}{mc[m]['mean']:>10.4f}{mc[m]['mc_se']:>9.4f}{mc[m]['bias']:>10.4f}"
        )
    print("  单次展示表（独立种子 20240601）:")
    for row in single:
        se = f"{row['se']:.3f}" if row["se"] is not None else "  -"
        print(f"    {row['method']:<22}{row['theta']:>10.4f}{se:>8}")
    return payload


if __name__ == "__main__":
    run_race(Config())
