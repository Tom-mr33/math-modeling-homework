"""交付前验收：按方案设定的误差标准逐项自动检查。

运行：在项目根目录执行本脚本；全项通过时输出 PASS，任一项失败则以非零状态退出。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import R_BASELINE, R_CONFORMAL, R_MC, R_RACE, R_SENSITIVITY, ROOT  # noqa: E402

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load(rel: str) -> dict:
    """rel 相对 results/ 根，如 'baseline/step4_fwl.json'。"""
    return json.loads((ROOT / "results" / rel).read_text(encoding="utf-8"))


def main() -> int:
    print("=" * 64)
    print("交付前验收（逐项比对预设误差标准）")
    print("=" * 64)

    # 1. FWL 数值验证
    s4 = load("baseline/step4_fwl.json")
    check("FWL ≡ 多元回归 (|Δ|<1e-8)", s4["abs_diff"] < 1e-8,
          f"|Δ| = {s4['abs_diff']:.2e}")

    # 2. 偏相关 t 等价性
    s5 = load("baseline/step5_partial_t.json")
    check("偏相关手算 t ≡ 软件 t (|Δ|<1e-6)", s5["abs_diff"] < 1e-6,
          f"|Δ| = {s5['abs_diff']:.2e}")

    # 3. conformal 覆盖率
    cf = load("conformal/conformal.json")
    check("conformal 经验覆盖 ≥ 名义−2pp", cf["coverage_ok"],
          f"名义 {cf['nominal']:.0%}, 经验 {cf['mean_coverage']:.1%} ({cf['reps']} 次)")

    # 4. MC 与 OVB 渐近理论一致
    ovb = load("mc/ovb_mc.json")
    check("MC: simple 均值 ≈ OVB 渐近 1.45 (z<3)", ovb["consistent_with_theory"],
          f"MC 均值 {ovb['mc_mean_simple']:.4f}, 理论 {ovb['theory_asymptotic_beta1_simple']:.4f}, z={ovb['z_simple_vs_theory']:.2f}")
    z_multi = abs(ovb["mc_mean_multiple"] - ovb["truth_beta1"]) / ovb["se_of_mc_mean_multiple"]
    check("MC: multiple 均值 ≈ 真值 2 (z<3)", z_multi < 3,
          f"MC 均值 {ovb['mc_mean_multiple']:.4f}, z={z_multi:.2f}")

    # 5. CH2020 恒等式验证
    sens = load("sensitivity/sensitivity.json")
    v = sens["validation_drop_x3"]
    check("CH2020 删X3找回 t (|Δ|<1e-6)", v["exact"],
          f"|Δ| = {v['abs_diff']:.2e}; RV(显著)={sens['rv_significance']:.4f}")

    # 6. 估计器竞赛：DML-RF 显著 closer to truth than OLS
    race = load("race/race.json")
    ols_bias = abs(race["mc"]["OLS(线性)"]["bias"])
    rf_bias = abs(race["mc"]["DML-RF"]["bias"])
    check("竞赛: DML-RF 偏差 < 线性 OLS 偏差", rf_bias < ols_bias,
          f"OLS |偏差|={ols_bias:.3f} vs DML-RF |偏差|={rf_bias:.3f} (n={race['n']})")

    # 7. 参数灵敏度：MC 偏误与解析预测一致
    ps = load("robustness/param_sensitivity.json")
    devs = [abs(r["bias_simple"] - (r["plim_simple_theory"] - 2)) for r in ps["rows"] if r["sigma"] == 5.0]
    check("灵敏度: 36 组网格上 MC 偏误 ≡ 解析式", max(devs) < 0.12 and sum(devs) / len(devs) < 0.06,
          f"最大偏离 {max(devs):.3f}, 平均 {sum(devs)/len(devs):.3f}")

    # 8. 稳健性：异方差下 HC1 覆盖率优于经典 SE
    rb = load("robustness/inference_robustness.json")["coverage"]
    hetero = rb["异方差"]
    check("稳健性: 异方差下 HC1 覆盖率 > 经典 SE", hetero["coverage_hc1"] > hetero["coverage_classic"],
          f"经典 {hetero['coverage_classic']:.3f} vs HC1 {hetero['coverage_hc1']:.3f}")

    # 一键复现：主流程二次运行数值不变
    import hashlib

    def results_digest() -> str:
        h = hashlib.sha256()
        for f in sorted(R_BASELINE.glob("step*.json")):
            h.update(f.read_bytes())
        return h.hexdigest()

    d1 = results_digest()
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "main.py")],
        cwd=ROOT, capture_output=True, check=True,
    )
    d2 = results_digest()
    check("一键复现: 主流程重跑数值摘要不变", d1 == d2, f"digest {d1[:12]}… == {d2[:12]}…")

    # 8. pytest 全部通过
    r = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", "--no-header"],
        cwd=ROOT, capture_output=True, text=True,
    )
    passed = "passed" in r.stdout and "failed" not in r.stdout
    check("单元测试 pytest 全部通过", passed and r.returncode == 0,
          r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[:80])

    n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
    print("=" * 64)
    print(f"验收结果：{len(CHECKS) - n_fail}/{len(CHECKS)} 通过" + ("，全部 PASS" if n_fail == 0 else "，存在 FAIL"))
    print("=" * 64)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
