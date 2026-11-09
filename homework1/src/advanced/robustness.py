"""advanced/robustness.py —— 第六章 6.1 参数灵敏度分析 与 6.2 统计推断稳健性检验。

6.1 参数灵敏度：在 n ∈ {30,60,120,300}、σ ∈ {2,5,10}、ρ12 ∈ {-0.2,-0.5,-0.8} 的
    网格上重复抽样，考察 (i) 简单回归系数偏误是否与解析预测一致；(ii) 检验功效随样本量
    的变化。M=200/组。
6.2 推断稳健性：在同方差、异方差、厚尾（t3）、含异常值四种误差结构下比较
    经典 SE / HC1 / Bootstrap 百分位法对 β1 的 95% 置信区间覆盖率（R=300）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import R_ROBUSTNESS, Config, ensure_dirs, rng_for  # noqa: E402
from datagen import generate_dataset  # noqa: E402
from ols import ols_fit  # noqa: E402

NUM_COLS = ["X1", "X2", "X3"]


def _beta1_with_inference(y: np.ndarray, X: np.ndarray) -> tuple[float, float, float]:
    """返回 (β̂1, 经典 SE, p 值)。"""
    res = ols_fit(y, X, add_intercept=True)
    se = np.sqrt(res.rss / res.df_resid * np.diag(res.xtx_inv))
    t = res.beta[1] / se[1]
    p = 2 * stats.t.sf(abs(t), res.df_resid)
    return float(res.beta[1]), float(se[1]), float(p)


def _simple_beta1(y: np.ndarray, x1: np.ndarray) -> tuple[float, float, float]:
    """简单回归的 (β̂1, SE, p)。"""
    X = x1[:, None]
    res = ols_fit(y, X, add_intercept=True)
    se = np.sqrt(res.rss / res.df_resid * np.diag(res.xtx_inv))
    t = res.beta[1] / se[1]
    p = 2 * stats.t.sf(abs(t), res.df_resid)
    return float(res.beta[1]), float(se[1]), float(p)


def _cov_with_rho(rho: float) -> tuple:
    """变量顺序 (X2,X3,X1)：Cov(X2,X1)=rho，其余按原设定。"""
    return ((1.0, 0.3, rho), (0.3, 1.0, -0.2), (rho, -0.2, 1.0))


def run_param_sensitivity(reps: int = 200) -> dict:
    ensure_dirs()
    base = Config()
    ns = [30, 60, 120, 300]
    sigmas = [2.0, 5.0, 10.0]
    rhos = [-0.2, -0.5, -0.8]
    rows = []
    for rho in rhos:
        for n in ns:
            for sigma in sigmas:
                cfg = Config(n=n, cov=_cov_with_rho(rho), sigma_eps=sigma)
                rng = rng_for(cfg, "paramsens")
                seeds = rng.integers(0, 2**31, size=reps)
                b_simple, b_multi, rej_simple, rej_multi = [], [], 0, 0
                for s in seeds:
                    df = generate_dataset(cfg, np.random.default_rng(int(s)))
                    y = df["Y"].to_numpy()
                    x1 = df["X1"].to_numpy()
                    X = df[NUM_COLS].to_numpy()
                    bs, _, ps = _simple_beta1(y, x1)
                    bm, _, pm = _beta1_with_inference(y, X)
                    b_simple.append(bs)
                    b_multi.append(bm)
                    rej_simple += int(ps < 0.05)
                    rej_multi += int(pm < 0.05)
                plim_simple = base.beta_true[0] + base.beta_true[1] * rho + base.beta_true[2] * (-0.2)
                rows.append(
                    {
                        "rho12": rho,
                        "n": n,
                        "sigma": sigma,
                        "plim_simple_theory": plim_simple,
                        "mean_simple": float(np.mean(b_simple)),
                        "bias_simple": float(np.mean(b_simple) - base.beta_true[0]),
                        "mean_multi": float(np.mean(b_multi)),
                        "bias_multi": float(np.mean(b_multi) - base.beta_true[0]),
                        "reject_simple": rej_simple / reps,
                        "reject_multi": rej_multi / reps,
                    }
                )

    payload = {"reps": reps, "grid": {"n": ns, "sigma": sigmas, "rho12": rhos}, "rows": rows}
    out = R_ROBUSTNESS / "param_sensitivity.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 控制台摘要：固定 sigma=5 的二维切片
    print(f"[6.1 参数灵敏度] M={reps}/组，共 {len(rows)} 组")
    print(f"  {'rho':>5}{'n':>5}{'σ':>5}{'bias_simple':>13}{'理论':>9}{'bias_multi':>12}{'拒绝率multi':>12}")
    for r in rows:
        if r["sigma"] == 5.0:
            print(
                f"  {r['rho12']:>5.1f}{r['n']:>5}{r['sigma']:>5.0f}"
                f"{r['bias_simple']:>13.3f}{r['plim_simple_theory']-2:>9.3f}"
                f"{r['bias_multi']:>12.3f}{r['reject_multi']:>12.2f}"
            )
    return payload


# ---------------- 6.2 推断稳健性 ----------------

def _gen_scenario(kind: str, cfg: Config, rng: np.random.Generator):
    """按误差结构生成数据：(Y, X1, X2, X3)。"""
    X = rng.multivariate_normal(np.asarray(cfg.mu), np.asarray(cfg.cov), size=cfg.n)
    X2, X3, X1 = X[:, 0], X[:, 1], X[:, 2]
    b1, b2, b3 = cfg.beta_true
    lin = cfg.beta0_true + b1 * X1 + b2 * X2 + b3 * X3
    if kind == "同方差正态":
        eps = rng.normal(0, cfg.sigma_eps, cfg.n)
    elif kind == "异方差":
        eps = rng.normal(0, 1, cfg.n) * cfg.sigma_eps * (1 + 0.4 * np.abs(X1 - X1.mean()))
    elif kind == "厚尾 t3":
        eps = rng.standard_t(3, cfg.n) * cfg.sigma_eps / np.sqrt(3)
    elif kind == "含 5% 异常值":
        eps = rng.normal(0, cfg.sigma_eps, cfg.n)
        idx = rng.choice(cfg.n, size=max(1, int(0.05 * cfg.n)), replace=False)
        eps[idx] += rng.normal(0, 20, len(idx))
    else:
        raise ValueError(kind)
    return lin + eps, X1, X2, X3


def _hc1_se(y: np.ndarray, X: np.ndarray, res) -> np.ndarray:
    bread = res.xtx_inv
    meat = (X.T * res.resid**2) @ X
    return np.sqrt(np.diag(bread @ meat @ bread * (res.n / res.df_resid)))


def run_inference_robustness(reps: int = 300, boot: int = 199) -> dict:
    ensure_dirs()
    cfg = Config()
    truth = cfg.beta_true[0]
    crit = stats.t.ppf(0.975, cfg.n - 4)  # 三种方法统一临界值，保证比较公平
    scenarios = ["同方差正态", "异方差", "厚尾 t3", "含 5% 异常值"]
    out = {}
    rng = rng_for(cfg, "robust")
    seeds = rng.integers(0, 2**31, size=(len(scenarios), reps))
    for k, kind in enumerate(scenarios):
        cov_classic = cov_hc1 = cov_boot = 0
        for s in seeds[k]:
            r = np.random.default_rng(int(s))
            y, X1, X2, X3 = _gen_scenario(kind, cfg, r)
            X = np.column_stack([X1, X2, X3])
            res = ols_fit(y, X, add_intercept=True)
            b = res.beta[1]
            se_c = float(np.sqrt(res.rss / res.df_resid * res.xtx_inv[1, 1]))
            se_h = float(_hc1_se(y, np.column_stack([np.ones(cfg.n), X]), res)[1])
            cov_classic += int(abs(b - truth) <= crit * se_c)
            cov_hc1 += int(abs(b - truth) <= crit * se_h)

            # Bootstrap 百分位区间
            boots = np.empty(boot)
            n = cfg.n
            for t in range(boot):
                idx = r.integers(0, n, n)
                rb = ols_fit(y[idx], X[idx], add_intercept=True)
                boots[t] = rb.beta[1]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            cov_boot += int(lo <= truth <= hi)
        out[kind] = {
            "coverage_classic": cov_classic / reps,
            "coverage_hc1": cov_hc1 / reps,
            "coverage_bootstrap": cov_boot / reps,
        }

    payload = {"reps": reps, "boot": boot, "nominal": 0.95, "coverage": out}
    path = R_ROBUSTNESS / "inference_robustness.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[6.2 推断稳健性] 名义覆盖 95%，R={reps}，Bootstrap B={boot}")
    print(f"  {'误差结构':<12}{'经典SE':>10}{'HC1':>9}{'Bootstrap':>11}")
    for kind, v in out.items():
        print(
            f"  {kind:<12}{v['coverage_classic']:>10.3f}"
            f"{v['coverage_hc1']:>9.3f}{v['coverage_bootstrap']:>11.3f}"
        )
    return payload


if __name__ == "__main__":
    run_param_sensitivity()
    run_inference_robustness()
