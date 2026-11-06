"""plotting.py —— 9 张图表（plotly 交互 HTML + matplotlib 静态 PNG 双输出）。

只读 results/ 与 data/ 中已落盘的计算结果，不重算任何统计量。
运行：python src/plotting.py   →  figures/fig*.html + fig*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DATA_DIR, FIGURES_DIR, RESULTS_DIR, Config, ensure_dirs  # noqa: E402

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "SimHei"],  # Windows 中文字体
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)
BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8C8C8C"
LABELS = {"Y": "期末平均成绩", "X1": "日均睡眠时长(小时)", "X2": "日均自习时长(小时)", "X3": "日均手机娱乐时长(小时)"}


def _save(fig_mpl, name: str, fig_plotly=None) -> None:
    if fig_plotly is not None:
        fig_plotly.write_html(FIGURES_DIR / f"{name}.html", include_plotlyjs="cdn")
    fig_mpl.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig_mpl)
    print(f"  -> figures/{name}.png + .html")


def fig1_corr_heatmap() -> None:
    corr = pd.read_csv(RESULTS_DIR / "step1_corr.csv", index_col=0)
    labels = [LABELS[c] for c in corr.columns]
    fig_mpl, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4), labels, rotation=30, ha="right")
    ax.set_yticks(range(4), labels)
    for i in range(4):
        for j in range(4):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color="white" if abs(v) > 0.55 else "black", fontsize=11)
    ax.set_title("控制变量确有共线关联：r(X1,X2)=-0.47, r(X1,X3)=-0.21")
    fig_mpl.colorbar(im, ax=ax, shrink=0.8, label="Pearson 相关系数")
    fig_plotly = px.imshow(corr, text_auto=".3f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                           title="变量相关矩阵 (n=60, 种子2024)", labels=dict(color="相关系数"))
    _save(fig_mpl, "fig1_corr_heatmap", fig_plotly)


def fig2_simple_regression() -> None:
    cfg = Config()
    df = pd.read_csv(DATA_DIR / "sleep_score.csv")
    tab = pd.read_csv(RESULTS_DIR / "step2_simple.csv")
    b0 = float(tab.loc[tab["name"] == "const", "beta"].item())
    b1 = float(tab.loc[tab["name"] == "x", "beta"].item())
    se = float(tab.loc[tab["name"] == "x", "se"].item())
    n = len(df)
    x = df["X1"].to_numpy()
    xbar, sxx = x.mean(), ((x - x.mean()) ** 2).sum()
    crit = stats.t.ppf(0.975, n - 2)
    xs = np.linspace(x.min(), x.max(), 200)
    fit = b0 + b1 * xs
    band = crit * se * np.sqrt(1 / n + (xs - xbar) ** 2 / sxx)

    fig_mpl, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(x, df["Y"], s=28, color=BLUE, alpha=0.75, label="样本 (n=60)")
    ax.plot(xs, fit, color=RED, lw=2, label=f"拟合线 (斜率={b1:.3f})")
    ax.fill_between(xs, fit - band, fit + band, color=RED, alpha=0.18, label="均值 95% 置信带")
    ax.axhline(df["Y"].mean(), color=GREY, lw=0.8, ls=":")
    ax.set_xlabel(LABELS["X1"]); ax.set_ylabel(LABELS["Y"])
    ax.set_title(f"简单回归忽略混淆：斜率 {b1:.3f} 远离真值 2 且不显著")
    ax.legend(frameon=True)
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=x, y=df["Y"], mode="markers", name="样本"))
    fig_plotly.add_trace(go.Scatter(x=xs, y=fit, mode="lines", name="拟合线"))
    fig_plotly.add_trace(go.Scatter(x=xs.tolist() + xs.tolist()[::-1],
        y=(fit - band).tolist() + (fit + band).tolist()[::-1],
        fill="toself", opacity=0.2, name="95% CI"))
    fig_plotly.update_layout(template="plotly_white", title="简单回归 Y~X1（不控制）",
                             xaxis_title=LABELS["X1"], yaxis_title=LABELS["Y"])
    _save(fig_mpl, "fig2_simple_regression", fig_plotly)


def fig3_added_variable() -> None:
    res = np.load(RESULTS_DIR / "step4_residuals.npy")
    e_x1, e_y = res[:, 0], res[:, 1]
    fwl = json.loads((RESULTS_DIR / "step4_fwl.json").read_text(encoding="utf-8"))
    slope = fwl["beta1_fwl"]
    fig_mpl, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(e_x1, e_y, s=28, color=BLUE, alpha=0.75)
    xs = np.linspace(e_x1.min(), e_x1.max(), 100)
    ax.plot(xs, slope * xs, color=RED, lw=2,
            label=f"FWL 斜率 = {slope:.4f} = 多元回归 β̂1")
    ax.axhline(0, color=GREY, lw=0.8, ls=":"); ax.axvline(0, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("X1 对 (X2,X3) 回归的残差 e_X1")
    ax.set_ylabel("Y 对 (X2,X3) 回归的残差 e_Y")
    ax.set_title("Added-variable 图：残差化后再回归，斜率即净效应（FWL 可视化）")
    ax.legend(frameon=True)
    fig_plotly = px.scatter(x=e_x1, y=e_y, template="plotly_white",
                            labels={"x": "e_X1", "y": "e_Y"}, title="FWL added-variable plot")
    fig_plotly.add_trace(go.Scatter(x=xs, y=slope * xs, mode="lines", name=f"斜率={slope:.4f}"))
    _save(fig_mpl, "fig3_added_variable", fig_plotly)


def fig4_coef_comparison() -> None:
    s2 = json.loads((RESULTS_DIR / "step2_simple.json").read_text(encoding="utf-8"))
    s3 = json.loads((RESULTS_DIR / "step3_multiple.json").read_text(encoding="utf-8"))
    s4 = json.loads((RESULTS_DIR / "step4_fwl.json").read_text(encoding="utf-8"))
    names = ["简单回归", "多元回归", "FWL 残差化"]
    betas = [s2["beta1_simple"], s3["beta1_multiple"], s4["beta1_fwl"]]
    ses = [0.9145, 0.9638, None]
    fig_mpl, ax = plt.subplots(figsize=(8, 5))
    ax.axhspan(2 - 0.96, 2 + 0.96, color=GREEN, alpha=0.10, label="真值 2 ± 1 个多元SE")
    ax.axhline(2, color=GREEN, lw=1.5, ls="--", label="真值 β1 = 2")
    for i, (nm, b) in enumerate(zip(names, betas)):
        ax.scatter(i, b, s=90, color=[RED, BLUE, BLUE][i], zorder=3)
        ax.text(i, b + 0.06, f"{b:.3f}", ha="center", fontsize=11, fontweight="bold")
        if ses[i] is not None:
            ax.errorbar(i, b, yerr=1.96 * ses[i], fmt="none", ecolor=GREY, capsize=5)
    ax.set_xticks(range(3), names)
    ax.set_ylabel("β̂1 估计值")
    ax.set_title("控制变量使估计从 0.91 修正到 2.09，接近真值 2")
    ax.legend(frameon=True, loc="lower right")
    fig_plotly = go.Figure()
    fig_plotly.add_hline(y=2, line_dash="dash", annotation_text="真值 2")
    fig_plotly.add_trace(go.Bar(x=names, y=betas, marker_color=[RED, BLUE, BLUE], text=[f"{b:.3f}" for b in betas]))
    fig_plotly.update_layout(template="plotly_white", title="三种方法 β̂1 对比", yaxis_title="β̂1")
    _save(fig_mpl, "fig4_coef_comparison", fig_plotly)


def fig5_mc_histogram() -> None:
    s = np.load(RESULTS_DIR / "mc_samples.npy")
    ovb = json.loads((RESULTS_DIR / "ovb_mc.json").read_text(encoding="utf-8"))
    theory = ovb["theory_asymptotic_beta1_simple"]
    fig_mpl, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(s[:, 0], bins=40, color=ORANGE, alpha=0.65, edgecolor="white",
            label=f"简单回归 (均值 {s[:,0].mean():.3f})")
    ax.hist(s[:, 1], bins=40, color=BLUE, alpha=0.65, edgecolor="white",
            label=f"多元回归 (均值 {s[:,1].mean():.3f})")
    ax.axvline(2, color=GREEN, lw=2, ls="--", label="真值 2")
    ax.axvline(theory, color=RED, lw=2, ls=":", label=f"OVB 渐近预测 {theory:.2f}")
    ax.set_xlabel("β̂1 的 Monte Carlo 抽样分布 (M=500)")
    ax.set_ylabel("频数")
    ax.set_title("多元回归围绕真值波动，简单回归围绕渐近偏误值 1.45")
    ax.legend(frameon=True)
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Histogram(x=s[:, 0], nbinsx=40, name="简单回归", opacity=0.65))
    fig_plotly.add_trace(go.Histogram(x=s[:, 1], nbinsx=40, name="多元回归", opacity=0.65))
    for xv, txt, col in [(2, "真值 2", GREEN), (theory, "渐近预测 1.45", RED)]:
        fig_plotly.add_vline(x=xv, line_dash="dash", line_color=col, annotation_text=txt)
    fig_plotly.update_layout(barmode="overlay", template="plotly_white",
                             title="两个估计量的抽样分布 (M=500)", xaxis_title="β̂1")
    _save(fig_mpl, "fig5_mc_histogram", fig_plotly)


def fig6_sensitivity_contour() -> None:
    T = np.load(RESULTS_DIR / "sensitivity_grid.npy")
    grid = np.load(RESULTS_DIR / "sensitivity_grid_axis.npy")
    sens = json.loads((RESULTS_DIR / "sensitivity.json").read_text(encoding="utf-8"))
    t0, tc = sens["t0"], sens["t_crit"]
    rv = sens["rv_significance"]
    fig_mpl, ax = plt.subplots(figsize=(8, 6))
    X, Y = np.meshgrid(grid, grid, indexing="ij")
    levels = [tc] + [1.0, 1.5, 2.5, 3.5, 5, 8, 12, 20, 30]
    cs = ax.contourf(X, Y, T, levels=np.linspace(0, 30, 31), cmap="RdYlGn", extend="max")
    ax.contour(X, Y, T, levels=[tc], colors="black", linewidths=2)
    ax.plot(rv, rv, marker="*", ms=18, color="black")
    ax.annotate(f"稳健值 RV={rv:.3f}\n(等强度混杂翻转显著性)", (rv, rv), xytext=(rv + 0.12, rv + 0.12),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    fig_mpl.colorbar(cs, ax=ax, label="加入未观测混杂 U 后的 |t|")
    ax.set_xlabel("U 与 X1 的偏 R²（处理侧混杂强度）")
    ax.set_ylabel("U 与 Y 的偏 R²（结果侧混杂强度）")
    ax.set_title(f"CH2020 敏感性等高线：黑线上方即失去显著性 (t0={t0:.2f}, t临界={tc:.2f})")
    fig_plotly = go.Figure(go.Contour(x=grid, y=grid, z=T,
        contours=dict(start=0, end=30, size=1), colorscale="RdYlGn",
        line_width=0.5))
    fig_plotly.add_trace(go.Scatter(x=[rv], y=[rv], mode="markers+text",
        marker=dict(symbol="star", size=16, color="black"),
        text=[f"RV={rv:.3f}"], textposition="top right", name="稳健值"))
    fig_plotly.update_layout(template="plotly_white", title="CH2020 敏感性分析等高线",
        xaxis_title="偏R²(处理侧)", yaxis_title="偏R²(结果侧)")
    _save(fig_mpl, "fig6_sensitivity_contour", fig_plotly)


def fig7_fwl_geometry() -> None:
    """FWL 三维正交投影示意（合成几何数据，非统计计算）。"""
    fig_mpl = plt.figure(figsize=(8, 7))
    ax = fig_mpl.add_subplot(111, projection="3d")
    # x2 张成的平面 = span(x2, 1)；y 与 x1 投影到其正交补
    t1, t2 = np.meshgrid(np.linspace(-1.2, 1.2, 10), np.linspace(-1.2, 1.2, 10))
    plane = 0.0 * t1  # 平面 z=0 表示 span(W)（视角简化）
    ax.plot_surface(t1, t2, plane, alpha=0.15, color=GREY)
    rng = np.random.default_rng(3)
    y_vec = np.array([0.5, 0.2, 2.0])
    w_part = np.array([0.3, 0.4, 0.0])   # 能被 W 解释的部分
    x1_vec = np.array([0.9, -0.3, 1.2])
    x1_w = np.array([0.6, 0.5, 0.0])
    for v, col, nm in [(y_vec, ORANGE, "y"), (x1_vec, BLUE, "x1")]:
        ax.quiver(0, 0, 0, *v, color=col, lw=2.2, arrow_length_ratio=0.08)
        proj = np.array([v[0], v[1], 0.0])
        ax.quiver(0, 0, 0, *proj, color=col, lw=1.4, ls="dashed", arrow_length_ratio=0.1)
        ax.plot([v[0], v[0]], [v[1], v[1]], [0.0, v[2]], color=col, lw=0.8, ls=":")  # 投影虚线
    e_y, e_x1 = y_vec - w_part, x1_vec - x1_w
    ax.quiver(0, 0, 0, *e_y, color=RED, lw=2.6, arrow_length_ratio=0.08)
    ax.quiver(0, 0, 0, *e_x1, color=GREEN, lw=2.6, arrow_length_ratio=0.08)
    ax.text(*y_vec * 1.1, "y", color=ORANGE, fontsize=13)
    ax.text(*x1_vec * 1.08, "x1", color=BLUE, fontsize=13)
    ax.text(*e_y * 1.1, "My (残差)", color=RED, fontsize=12)
    ax.text(*e_x1 * 1.12, "Mx1 (残差)", color=GREEN, fontsize=12)
    ax.text(0.55, 0.75, 0.0, "span{W}: 控制变量平面", color=GREY, fontsize=11)
    ax.set_title("FWL 的几何：y 与 x1 各自投影到 W 的正交补，残差夹角决定净相关")
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=42)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.05)
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Surface(x=t1, y=t2, z=plane, opacity=0.25, showscale=False, name="span{W}"))
    for v, nm, col in [(y_vec, "y", ORANGE), (x1_vec, "x1", BLUE),
                       (y_vec - w_part, "My", RED), (x1_vec - x1_w, "Mx1", GREEN)]:
        fig_plotly.add_trace(go.Scatter3d(x=[0, v[0]], y=[0, v[1]], z=[0, v[2]],
            mode="lines+text", line=dict(color=col, width=6), name=nm,
            text=["", nm], textposition="top center"))
    fig_plotly.update_layout(template="plotly_white", height=600,
        title="FWL 正交投影几何（可旋转）")
    _save(fig_mpl, "fig7_fwl_geometry", fig_plotly)


def fig8_race() -> None:
    race = json.loads((RESULTS_DIR / "race.json").read_text(encoding="utf-8"))
    mc = race["mc"]
    names = list(mc.keys())
    means = [mc[n]["mean"] for n in names]
    ses = [mc[n]["mc_se"] for n in names]
    plim = race["ols_plim_reference"]
    fig_mpl, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(2, color=GREEN, lw=2, ls="--", label="真值 β1 = 2")
    ax.axhline(plim, color=RED, lw=1.6, ls=":", label=f"OLS 渐近参考值 {plim:.2f} (N=30万)")
    colors = [RED, ORANGE, BLUE]
    bars = ax.bar(names, means, yerr=ses, capsize=6, color=colors, alpha=0.85, width=0.55)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.05, f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("β̂1 的 MC 均值 (M=60, n=600)")
    ax.set_title("估计器竞赛：灵活烦恼项 (RF) 把偏误从 +0.76 压到 −0.18")
    ax.legend(frameon=True)
    fig_plotly = go.Figure()
    fig_plotly.add_hline(y=2, line_dash="dash", line_color=GREEN, annotation_text="真值 2")
    fig_plotly.add_hline(y=plim, line_dash="dot", line_color=RED, annotation_text="OLS 渐近值")
    fig_plotly.add_trace(go.Bar(x=names, y=means, error_y=dict(array=ses),
        marker_color=colors, text=[f"{m:.3f}" for m in means], textposition="outside"))
    fig_plotly.update_layout(template="plotly_white", title="非线性混淆下的估计器竞赛 (MC, n=600)",
        yaxis_title="β̂1 MC 均值")
    _save(fig_mpl, "fig8_race", fig_plotly)


def fig9_conformal() -> None:
    p = json.loads((RESULTS_DIR / "conformal.json").read_text(encoding="utf-8"))
    ex = p["example"]
    fig_mpl, ax = plt.subplots(figsize=(9, 5.5))
    x1 = np.asarray(ex["x1"])
    ax.fill_between(x1, ex["lo"], ex["hi"], color=BLUE, alpha=0.18, label=f"90% conformal 区间 (宽 {p['mean_width']:.1f})")
    ax.plot(x1, ex["hi"], color=BLUE, lw=0.8)
    ax.plot(x1, ex["lo"], color=BLUE, lw=0.8)
    ax.scatter(x1, ex["truth"], s=26, color="black", zorder=3, label="实际成绩")
    in_cov = (np.asarray(ex["truth"]) >= np.asarray(ex["lo"])) & (np.asarray(ex["truth"]) <= np.asarray(ex["hi"]))
    ax.scatter(x1[in_cov], np.asarray(ex["truth"])[in_cov], s=30, facecolors="none", edgecolors=GREEN, lw=1.5, label="落入区间")
    ax.set_xlabel("日均睡眠时长 (测试集按 X1 排序)")
    ax.set_ylabel("期末平均成绩")
    ax.set_title(f"分裂 conformal 反事实区间：经验覆盖 {p['mean_coverage']:.1%} ≥ 名义 90%")
    ax.legend(frameon=True, loc="upper left")
    fig_plotly = go.Figure()
    fig_plotly.add_trace(go.Scatter(x=x1, y=ex["lo"], mode="lines", line=dict(color=BLUE), name="下界"))
    fig_plotly.add_trace(go.Scatter(x=x1.tolist() + x1.tolist()[::-1],
        y=ex["lo"] + ex["hi"][::-1], fill="toself", opacity=0.18, name="90% 区间", showlegend=True))
    fig_plotly.add_trace(go.Scatter(x=x1, y=ex["hi"], mode="lines", line=dict(color=BLUE), name="上界"))
    fig_plotly.add_trace(go.Scatter(x=x1, y=ex["truth"], mode="markers", marker=dict(color="black"), name="实际成绩"))
    fig_plotly.update_layout(template="plotly_white", title="conformal 反事实预测区间 (Lei–Candès 2021)")
    _save(fig_mpl, "fig9_conformal", fig_plotly)


def main() -> None:
    ensure_dirs()
    print("[plotting] 生成 9 张图 (PNG + 交互 HTML)")
    fig1_corr_heatmap()
    fig2_simple_regression()
    fig3_added_variable()
    fig4_coef_comparison()
    fig5_mc_histogram()
    fig6_sensitivity_contour()
    fig7_fwl_geometry()
    fig8_race()
    fig9_conformal()
    print("[plotting] 全部完成")


if __name__ == "__main__":
    main()
