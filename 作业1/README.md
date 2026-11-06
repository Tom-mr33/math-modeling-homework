# 作业1：多元线性回归中"控制变量"的数学推证与随机仿真练习

> 完整技术方案见 [方案设计.md](./方案设计.md)：方法调研、理论推导链条、假设清单、创新扩展（OVB 定量验证、DML）、图表规划与代码架构。

## 题目背景

研究学生日均睡眠时长对期末平均成绩的影响。睡眠时长与自习时长、手机娱乐时长相互关联，
直接做简单回归会把其他变量的影响混入睡眠的系数中。需在控制 $X_2$（自习）、$X_3$（手机）
的条件下识别 $X_1$（睡眠）对 $Y$（成绩）的净影响，并用 FWL 残差化方法验证。

## 变量设定（n = 60）

| 变量 | 含义 | 角色 |
| ---- | ---- | ---- |
| $Y$ | 期末平均成绩 | 因变量 |
| $X_1$ | 日均睡眠时长（小时） | 关注的自变量 |
| $X_2$ | 日均自习时长（小时） | 控制变量 |
| $X_3$ | 日均手机娱乐时长（小时） | 控制变量 |

## 数据生成算法

- 随机种子：`2024`
- $(X_2, X_3, X_1)^T \sim N_3(\mu, \Sigma)$，$\mu = (3, 2, 3)^T$，
  $\Sigma = \begin{pmatrix} 1.0 & 0.3 & -0.5 \\ 0.3 & 1.0 & -0.2 \\ -0.5 & -0.2 & 1.0 \end{pmatrix}$
- $\varepsilon \sim N(0, 5^2)$
- 真实模型：$Y = 60 + 2X_1 + 1.5X_2 - 1X_3 + \varepsilon$（真值 $\beta_1 = 2$）
- 输出：`data/sleep_score.csv`

## 实施步骤

1. **数据生成与描述统计**：均值、标准差、4×4 相关矩阵，说明控制变量的共线关联
2. **简单回归（不控制）**：$Y \sim X_1$，报告系数/标准误/t/p/R²，解释偏差方向
3. **多元回归（控制）**：$Y \sim X_1 + X_2 + X_3$，检验 $H_0: \beta_1 = 0$
4. **FWL 残差化验证**：$e_Y$ 对 $e_{X_1}$ 无截距回归，验证与步骤 3 的 $\hat\beta_1$ 差 < 1e-8
5. **偏相关与 t 检验等价性**：$t^2 = (n-k)\frac{r_{1y\cdot 23}^2}{1-r_{1y\cdot 23}^2}$，k = 4，验证误差 < 1e-6
6. **对比讨论**：控制前后系数变化、遗漏变量偏误、相关 ≠ 因果
7. **（加分）仿真重复**：M = 500 次，画 $\hat\beta_1^{simple}$ 与 $\hat\beta_1^{multiple}$ 直方图

## 提交物清单

- [x] 数据文件 `data/sleep_score.csv`（种子 2024，n=60）
- [x] 可运行脚本 `src/`（`python src/main.py` 一键复现步骤 1–6，数值已验收）
- [x] 前沿扩展 `src/advanced/`（OVB 解析验证 / CH2020 敏感性分析 / DML / R-learner / DR-learner / CausalForest / conformal / MC 估计器竞赛）
- [x] 图表 `figures/` 9 张（plotly 交互 HTML + matplotlib PNG 双输出，`python src/plotting.py` 重新生成）
- [x] 验收 QA `src/qa_check.py`：9/9 通过（FWL 1e-13、t 等价 1e-13、conformal 覆盖 93.5%、CH2020 恒等式 3e-13、一键复现摘要一致、pytest 7/7）
- [ ] 报告 `report/`（八部分 + 前沿方法综述 + 附录）

## 快速开始

```bash
pip install -r requirements.txt   # econml 可选（缺失自动降级）
python src/main.py                # 步骤 1–6 流水线 → results/
python src/advanced/ovb.py        # MC 仿真 M=500 + OVB 渐近验证
python src/advanced/sensitivity.py # CH2020 敏感性分析
python src/advanced/race.py       # 非线性混淆估计器竞赛 (n=600, M=60)
python src/advanced/conformal.py  # conformal 覆盖率验证
python src/plotting.py            # 9 张图 → figures/
python src/qa_check.py            # 交付验收 (9 项)
python -m pytest tests/ -v        # 单元测试 (7 项)
```
