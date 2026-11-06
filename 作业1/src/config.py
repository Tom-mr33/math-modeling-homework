"""config.py —— 作业1唯一参数源。

所有超参数、路径与随机种子从这里读取，保证任何模块独立运行都可复现。
随机性管理：SeedSequence(seed).spawn(N) 按流名顺序派生子种子，
各模块通过 rng_for("流名") 获取互不重叠的随机流。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 项目根 = src/ 的上一级（作业1/），全 pathlib 解析，规避中文用户名编码问题
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
REPORT_DIR = ROOT / "report"


def ensure_dirs() -> None:
    for d in (DATA_DIR, FIGURES_DIR, RESULTS_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# 随机流名 -> spawn 索引（顺序即种子派生顺序，新增流只能追加在末尾）
STREAMS = ["datagen", "mc", "dml", "meta", "conformal", "nonlinear"]


@dataclass(frozen=True)
class Config:
    # --- 数据生成（作业第 3 节算法）---
    seed: int = 2024
    n: int = 60
    # 变量顺序 (X2, X3, X1)，与习题 Σ 矩阵一致
    mu: tuple[float, float, float] = (3.0, 2.0, 3.0)
    cov: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.3, -0.5),
        (0.3, 1.0, -0.2),
        (-0.5, -0.2, 1.0),
    )
    sigma_eps: float = 5.0
    # 真实模型 Y = beta0 + beta1*X1 + beta2*X2 + beta3*X3 + eps
    beta0_true: float = 60.0
    beta_true: tuple[float, float, float] = (2.0, 1.5, -1.0)  # (β1, β2, β3)

    # --- 仿真 ---
    mc_reps: int = 500

    # --- 统计 ---
    ci_level: float = 0.95

    # --- 派生 ---
    k: int = 4  # 参数个数 = 截距 + 3 个系数（用于自由度 n-k）

    @property
    def beta_full(self) -> np.ndarray:
        """完整真值向量 [β0, β1, β2, β3]。"""
        return np.array([self.beta0_true, *self.beta_true])


def make_config(**overrides) -> Config:
    return Config(**overrides)


def rng_for(cfg: Config, stream: str) -> np.random.Generator:
    """按流名派生子随机数生成器（确定性、互不重叠）。"""
    if stream not in STREAMS:
        raise KeyError(f"未知随机流 {stream!r}，可选：{STREAMS}")
    idx = STREAMS.index(stream)
    child = np.random.SeedSequence(cfg.seed).spawn(len(STREAMS))[idx]
    return np.random.default_rng(child)
