"""纯 numpy 的最小二乘核心：OLS / FWL / 偏相关。

不依赖 statsmodels（后者仅在 tests 中作互验基准）。
矩阵记号与方案设计文档第 2 节一致：
  OLS:      beta = (X'X)^{-1} X'y
  FWL:      beta1 = <M_Z x1, M_Z y> / ||M_Z x1||^2,  M_Z = I - Z(Z'Z)^{-1}Z'
  偏相关:   r_{1y.Z} = corr(M_Z x1, M_Z y)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OLSResult:
    beta: np.ndarray        # (k,) 系数向量
    fitted: np.ndarray      # (n,)
    resid: np.ndarray       # (n,)
    xtx_inv: np.ndarray     # (k,k) (X'X)^{-1}，供推断层复用
    rank: int
    n: int
    k: int                  # 参数个数（含截距列，若加入）

    @property
    def rss(self) -> float:
        return float(self.resid @ self.resid)

    @property
    def df_resid(self) -> int:
        return self.n - self.k

    @property
    def r_squared(self) -> float:
        """中心化 R²（ESS/TSS 分解）。仅对含截距的拟合有意义；
        本仓库所有 R² 调用均在 add_intercept=True 的模型上。"""
        y_bar = float(np.mean(self.fitted))  # 含截距时 fitted.mean() == y.mean()
        tss = float(((self.fitted - y_bar) ** 2).sum() + self.rss)
        return 1.0 - self.rss / tss

    @property
    def adj_r_squared(self) -> float:
        n, k = self.n, self.k
        return 1.0 - (1.0 - self.r_squared) * (n - 1) / (n - k)


def ols_fit(
    y: np.ndarray, X: np.ndarray, add_intercept: bool = False
) -> OLSResult:
    """无截距（默认）或带截距的 OLS。

    Parameters
    ----------
    y : (n,) 因变量
    X : (n,p) 设计矩阵
    add_intercept : True 时在最左列加一列 1
    """
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    xtx = X.T @ X
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (X.T @ y)
    fitted = X @ beta
    resid = y - fitted
    rank = int(np.linalg.matrix_rank(X))
    return OLSResult(beta, fitted, resid, xtx_inv, rank, len(y), X.shape[1])


def residualize(u: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """把 u 对控制变量矩阵 Z（含截距）回归，返回残差 M_Z u。

    M_Z u 即 FWL 定理中的残差化算子作用结果。
    """
    return ols_fit(u, Z, add_intercept=True).resid


def fwl_beta1(y: np.ndarray, x1: np.ndarray, Z: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """FWL 两步估计：e_y 对 e_x1 的无截距回归斜率。

    Returns
    -------
    (beta1_fwl, e_y, e_x1) —— 残差一并返回，供 added-variable plot 与偏相关使用。
    """
    e_y = residualize(y, Z)
    e_x1 = residualize(x1, Z)
    beta1 = float(e_x1 @ e_y / (e_x1 @ e_x1))
    return beta1, e_y, e_x1


def partial_correlation(y: np.ndarray, x1: np.ndarray, Z: np.ndarray) -> float:
    """控制 Z 后 x1 与 y 的偏相关系数 = 两残差的样本相关系数。"""
    e_y = residualize(y, Z)
    e_x1 = residualize(x1, Z)
    return float(
        e_x1 @ e_y / np.sqrt((e_x1 @ e_x1) * (e_y @ e_y))
    )


def t_from_partial_r2(r: float, n: int, k: int) -> float:
    """由偏相关直接构造 t 统计量：t^2 = (n-k) * r^2 / (1 - r^2)。

    与多元回归输出中 beta1 的 t 值在数值上等价（方案设计文档第 2 节第 4 条）。
    k 为回归参数总数（含截距）。
    """
    r2 = r * r
    return float(np.sqrt(n - k) * np.sqrt(r2 / (1.0 - r2)))
