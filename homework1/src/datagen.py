"""数据生成：按题设设定产生主样本与实验场景。

只负责生成数据并返回 DataFrame，不知道下游谁在消费。
变量顺序约定：外部协方差矩阵按 (X2, X3, X1) 给出，输出按 (Y, X1, X2, X3) 排列。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, Config, rng_for

COLUMNS = ["Y", "X1", "X2", "X3"]


def generate_dataset(cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """按题设设定生成一次数据。

    自变量取自联合正态分布，误差项独立同分布，因变量由线性模型给出；
    返回含 Y 与三个自变量的数据表。
    """
    X = rng.multivariate_normal(
        mean=np.asarray(cfg.mu), cov=np.asarray(cfg.cov), size=cfg.n
    )
    X2, X3, X1 = X[:, 0], X[:, 1], X[:, 2]
    eps = rng.normal(0.0, cfg.sigma_eps, size=cfg.n)
    b1, b2, b3 = cfg.beta_true
    Y = cfg.beta0_true + b1 * X1 + b2 * X2 + b3 * X3 + eps
    return pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3}, columns=COLUMNS)


def generate_nonlinear_dataset(
    cfg: Config,
    rng: np.random.Generator,
    g=np.sin,
    skew_x2: bool = True,
) -> pd.DataFrame:
    """扩展 3 的实验组：控制变量以非线性形式 g(X2) 进入 Y。

    skew_x2=True（默认）时对 X2 做单调指数变换使其右偏（均值/方差保持 μ/σ²）。
    这一步必不可少：联合正态设计下 X1|W 线性、残差与任何 g(W) 的线性投影
    残差正交，线性 OLS 对 beta1 天然无偏（数值验证 plim≈2.0）；
    只有破坏正态性（右偏的睡眠时长分布，现实中也更合理），
    线性误设才会真正传入 beta1，从而区分 OLS 与 DML 类方法。
    """
    X = rng.multivariate_normal(
        mean=np.asarray(cfg.mu), cov=np.asarray(cfg.cov), size=cfg.n
    )
    X2, X3, X1 = X[:, 0], X[:, 1], X[:, 2]
    if skew_x2:
        # 对潜在正态 Z2 做指数变换后标准化回 (mean=mu2, sd=1)，保持与 X1/X3 的秩相关
        z2_raw = X2
        s = np.exp(0.6 * z2_raw)
        X2 = (s - s.mean()) / s.std(ddof=0) + cfg.mu[0]
    eps = rng.normal(0.0, cfg.sigma_eps, size=cfg.n)
    b1, b2, b3 = cfg.beta_true
    Y = cfg.beta0_true + b1 * X1 + b2 * g(X2) + b3 * X3 + eps
    return pd.DataFrame({"Y": Y, "X1": X1, "X2": X2, "X3": X3}, columns=COLUMNS)


def save_csv(df: pd.DataFrame, filename: str = "sleep_score.csv") -> "Path":
    path = DATA_DIR / filename
    df.to_csv(path, index=False)
    return path


def load_main_dataset(cfg: Config, save: bool = True) -> pd.DataFrame:
    """生成（并落盘）主数据集：种子唯一确定，多次调用结果一致。"""
    df = generate_dataset(cfg, rng_for(cfg, "datagen"))
    if save:
        save_csv(df)
    return df
