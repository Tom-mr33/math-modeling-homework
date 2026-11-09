# 数学模型课程作业

重庆大学《数学模型》课程作业仓库，包含各次作业的习题、代码、数据与实验报告。

## 作业列表

| 作业 | 主题 | 状态 |
| ---- | ---- | ---- |
| [作业1](./作业1/) | 多元线性回归中"控制变量"的数学推证与随机仿真练习（FWL 定理验证） | 进行中 |

## 作业1：控制变量的数学推证与随机仿真

**研究问题**：在控制自习时长 $X_2$、手机娱乐时长 $X_3$ 的条件下，估计日均睡眠时长 $X_1$ 对期末成绩 $Y$ 的净效应。

- 总体模型：$Y = 60 + 2X_1 + 1.5X_2 - 1X_3 + \varepsilon,\quad \varepsilon \sim N(0, 5^2)$
- 样本量：$n = 60$，随机种子：`2024`

**核心内容**：

1. 理论推证：最小二乘代数解、Frisch–Waugh–Lovell 定理、偏相关系数与 t 检验的等价性
2. 随机仿真：生成 `sleep_score.csv`，描述统计与相关矩阵
3. 对比验证：简单回归 vs 多元回归 vs FWL 残差化回归，验证 $\hat\beta_1$ 数值一致（误差 < 1e-8）
4. （加分项）M = 500 次重复仿真，展示两种估计量的抽样分布

详见 [作业1/README.md](./作业1/README.md)。

## 环境依赖

- Python 3.10+
- NumPy、pandas、SciPy、statsmodels、matplotlib

```bash
pip install -r 作业1/requirements.txt
```

## 目录结构

```
数学模型作业/
├── README.md            # 本文件
├── LICENSE              # MIT License
└── 作业1/
    ├── 2026-11-习题.pdf  # 作业要求原文
    ├── README.md         # 作业1说明与进度
    ├── requirements.txt  # Python 依赖
    ├── src/              # 仿真与回归代码
    ├── data/             # sleep_score.csv（种子生成，入库）
    ├── tests/            # 数值互验测试
    ├── results/          # 数值结果落盘（json/csv，入库）
    ├── figures/          # 图表 PNG（本地生成，不入库）
    └── report/           # 实验报告（本地撰写，不入库）
```

> 报告与图表为本地运行产物（`python src/plotting.py` / `report/实验报告.md`），已加入 `.gitignore`，不在仓库中维护。

## License

[MIT](./LICENSE)
