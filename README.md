# FuturesModelFactory (期货模型工厂)

`FuturesModelFactory` 是一个基于 `polars` 的系统化量化因子开发与子系统治理框架。它的核心设计理念是**“防止研究过程的统计过拟合、强调数据的物源可追溯性、推行模型与因子的契约化治理”**。

它不仅仅是一个计算库，而是一整套限制研究员无序检索因子、规范策略表现审查的治理系统。

---

## 目录
- [整体图景与设计哲学](#整体图景与设计哲学)
- [核心机制解析](#核心机制解析)
  - [1. 因子与子系统注册制 (Registry)](#1-因子与子系统注册制-registry)
  - [2. 高性能因子诊断引擎 (Diagnostics)](#2-高性能因子诊断引擎-diagnostics)
  - [3. 对抗性冗余性检验 (Adversarial Redundancy)](#3-对抗性冗余性检验-adversarial-redundancy)
- [快速上手](#快速上手)
  - [环境准备](#环境准备)
  - [因子卡片 YAML 模板](#因子卡片-yaml-模板)
  - [代码使用示例](#代码使用示例)
- [开发与代码质量](#开发与代码质量)

---

## 整体图景与设计哲学

在系统化量化交易中，因子挖掘的经典失败路径是：**无边界的多维度因子搜索 → 统计虚假显著与多重共线性 → 实盘表现失效**。

`FuturesModelFactory` 通过以下“硬约束”重塑研究路径：
- **卡片备案制**：任何候选因子或特征子系统在进入策略回测和训练前，必须完成“卡片”注册。卡片不仅要求提供计算公式，还强制要求提供经济学假设、至少 3 个明确的失效条件、退市条件（Kill Conditions）以及不可剥离的逻辑合同。
- **高维对抗冗余评估**：新因子如果能被已有因子库通过非线性模型（如 Gradient Boosting）高精度重构（R2 超过阈值），即使它单独表现再好，也会被标记为“冗余”并拒绝入库。
- **物源追踪**：强制记录因子的搜索上下文，确保研究可复现。

---

## 核心机制解析

### 1. 因子与子系统注册制 (Registry)

系统通过 `FactorCard` 和 `SubsystemCard` 实行严格的元数据校验。

- **因子卡片 (`FactorCard`)**
  - **强制项**：`name` (名称)、`hypothesis` (经济学/市场行为学假设)、`mechanism_type` (机制类型)、`expected_sign` (预期符号)、至少 3 个 `failure_conditions` (失效场景描述)、`kill_conditions` (触发强平/停用条件)。
  - **安全审查**：拒绝模版中的 TBD / TODO 等占位符，促使研究员在写代码前想清楚研究逻辑。
- **子系统卡片 (`SubsystemCard`)**
  - 用于机器学习或深度表征的多因子集成子系统。
  - 强制定义 **Escape Dimensions（补偿维度）** 与 **Bottleneck Location（瓶颈层）**，确保表征模型不至于沦为无法监控的黑盒。

### 2. 高性能因子诊断引擎 (Diagnostics)

基于 `polars` 极速的截面处理能力，提供以下诊断：
- **IC & Rank IC 计算 (`ic.py`)**：支持日频或自定义时段的截面 Pearson/Rank IC 稳定度汇总（均值、标准差、ICIR、正 IC 占比等）。
- **IC 衰减分析 (`ic_decay.py`)**：追踪因子的预测能力随持仓周期（Lags）拉长后的衰减斜率。
- **单调性与分桶检验 (`monotonicity.py` / `buckets.py`)**：将因子在截面上进行等分桶，分析不同分桶的年化收益，计算单调性得分，识别非线性收益特征。
- **分布漂移检测 (`drift.py`)**：通过 KS 检验（Kolmogorov-Smirnov）或 PSI（Population Stability Index）实时检测候选因子的数据分布与回测区间的差异，自动触发调权或停用操作。

### 3. 对抗性冗余性检验 (Adversarial Redundancy)

对于新发掘的因子，系统除了进行常规的线性相关系数检查外，还内建了**对抗冗余分析**：
- **算法设计**：使用 `scikit-learn` 的梯度提升回归器 (`GradientBoostingRegressor`) 作为对抗者，尝试利用现有因子库预测新因子。
- **拒绝规则**：如果对抗模型的测试集 $R^2$ 超过该因子类别的阈值（例如量价因子为 `0.4`，基本面为 `0.6`），说明该因子只是已有因子的高维组合，并无独特增量信息，系统予以拒绝。

---

## 快速上手

### 环境准备

本项目推荐使用现代 Python 包管理器 `uv` 进行依赖安装和环境运行。

```bash
# 克隆仓库并初始化虚拟环境
uv venv
source .venv/bin/activate  # macOS / Linux

# 安装项目依赖
uv sync
```

### 因子卡片 YAML 模板

在 `registry` 目录中，因子应当以 YAML 文件的形式进行注册。例如 `factor_volume_momentum.yaml`：

```yaml
name: "factor_volume_momentum"
status: "candidate"
hypothesis: "在高成交量伴随价格突破时，资金动量因子具有正向预测力。"
mechanism_type: "trend_following"
expected_sign: "1"
inputs:
  - "close_price"
  - "volume"
failure_conditions:
  - { condition: "market_sideways", metric: "atr_ratio < 0.1" }
  - { condition: "low_liquidity", metric: "daily_volume < 1000" }
  - { condition: "high_correlation_with_index", metric: "beta_to_index > 1.2" }
kill_conditions:
  - "drawdown > 15%"
search_context:
  optimizer: "none"
  sample_period: "2020-01-01 to 2023-12-31"
deep_attribution:
  market_structure_parent: "volume_climax"
log_contrast_proxy:
  reference_factor: "volume_ma20"
```

### 代码使用示例

```python
import polars as pl
from futures_model_factory.registry.factor_cards import load_factor_card
from futures_model_factory.diagnostics.ic import information_coefficient, ic_summary
from futures_model_factory.diagnostics.redundancy import factor_redundancy_report

# 1. 加载并校验因子卡片
card = load_factor_card("path/to/factor_volume_momentum.yaml")
print(f"因子加载成功: {card.name}, 假设: {card.hypothesis}")

# 2. 准备数据 (Polars DataFrame)
# factors 包含: date, code, factor_value
factors_df = pl.DataFrame({
    "date": ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"],
    "code": ["IF2606", "IH2606", "IF2606", "IH2606"],
    "factor_value": [1.2, -0.5, 1.4, -0.2]
})

# labels 包含: date, code, next_ret (收益率标签)
labels_df = pl.DataFrame({
    "date": ["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-02"],
    "code": ["IF2606", "IH2606", "IF2606", "IH2606"],
    "next_ret": [0.012, -0.005, -0.002, 0.008]
})

# 3. 计算 IC 与 Rank IC
ic_by_date = information_coefficient(
    factors=factors_df,
    labels=labels_df,
    factor_name=card.name,
    label_col="next_ret"
)
print("每日 IC:")
print(ic_by_date)

# 4. 生成 IC 稳定性汇总
summary = ic_summary(ic_by_date)
print("IC 表现汇总:")
print(summary)
```

---

## 开发与代码质量

修改或扩展本项目代码时，请遵循以下规范：

- **代码风格检查**：使用 `ruff` 检查和修复格式
  ```bash
  uv run ruff check . --fix
  ```
- **格式化代码**：
  ```bash
  uv run ruff format .
  ```
- **运行测试**：
  ```bash
  uv run pytest
  ```
