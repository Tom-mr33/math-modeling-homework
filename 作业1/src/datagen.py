"""datagen.py —— 数据生成（作业要求 1 的算法）。

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
    """按习题步 2–4 生成一次数据。

    步 2: (X2, X3, X1)' ~ N3(mu, Sigma)
    步 3: eps ~ N(0, sigma^2)
    步 4: Y = 60 + 2*X1 + 1.5*X2 - 1*X3 + eps
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
    cfg: Config, rng: np.random.Generator, g=np.sin
) -> pd.DataFrame:
    """扩展 3 的实验组：控制变量以非线性形式 g(X2) 进入 Y。

    用于展示线性 OLS 的函数形式误设偏误，以及 DML/R-learner 的修正能力。
    其余设定与 generate_dataset 完全一致（同分布同系数）。
    """
    X = rng.multivariate_normal(
        mean=np.asarray(cfg.mu), cov=np.asarray(cfg.cov), size=cfg.n
    )
    X2, X3, X1 = X[:, 0], X[:, 1], X[:, 2]
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
