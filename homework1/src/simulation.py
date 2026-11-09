"""simulation.py —— Monte Carlo 引擎。

接收任意估计函数 (DataFrame) -> beta1，返回抽样分布样本与摘要。
OLS / FWL / DML / R-learner 共用同一引擎（方案设计文档解耦原则 2）。
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from config import Config, rng_for
from ols import ols_fit

Estimator = Callable[[pd.DataFrame], float]
DataGen = Callable[[np.random.Generator], pd.DataFrame]


def beta1_simple(df: pd.DataFrame) -> float:
    """步骤二估计量：Y ~ X1 的斜率。"""
    x = df["X1"].to_numpy()
    y = df["Y"].to_numpy()
    return float(x @ (y - y.mean()) / (x @ (x - x.mean())))


def beta1_multiple(df: pd.DataFrame) -> float:
    """步骤三估计量：Y ~ X1 + X2 + X3 的 beta1。"""
    res = ols_fit(
        df["Y"].to_numpy(), df[["X1", "X2", "X3"]].to_numpy(), add_intercept=True
    )
    return float(res.beta[1])


def run_mc(
    cfg: Config,
    estimator: Estimator,
    reps: int | None = None,
    stream: str = "mc",
    gen: DataGen | None = None,
) -> dict:
    """重复生成数据并记录估计量抽样分布。

    gen: 自定义数据生成器 (rng) -> DataFrame，默认线性模型 generate_dataset；
         非线性混淆场景传入 generate_nonlinear_dataset 的包装即可。
    每次重复用基础流派生的独立子种子，整体可复现。
    """
    from datagen import generate_dataset

    reps = reps or cfg.mc_reps
    base_rng = rng_for(cfg, stream)
    child_seeds = base_rng.integers(0, 2**31, size=reps)
    gen = gen or (lambda r: generate_dataset(cfg, r))

    samples = np.empty(reps)
    for m in range(reps):
        samples[m] = estimator(gen(np.random.default_rng(int(child_seeds[m]))))
    return {
        "samples": samples,
        "mean": float(samples.mean()),
        "std": float(samples.std(ddof=1)),
        "reps": reps,
    }
