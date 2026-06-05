# FuturesModelFactory

量化因子开发与子系统治理框架，旨在通过契约备案、高维对抗冗余评估和分布漂移监控，重塑系统化量化因子的研究与上线流程，防止统计过拟合。

## Language

**Factor (因子)**:
表示一种具备明确市场经济学假设、对未来资产收益率具有预测能力的数值特征信号。
_Avoid_: feature, alpha, signal

**Factor Card (因子卡片)**:
对候选因子在回测与训练前实行的强制性备案契约，包含其假设、失效条件和退市条件。
_Avoid_: specification, config file, yaml registry

**Subsystem Card (子系统卡片)**:
对多因子集成或机器学习表征子系统实行的物理结构与补偿维度约束契约。
_Avoid_: model card, ensemble config

**Adversarial Redundancy (对抗性冗余)**:
新因子可被已有因子库通过非线性监督模型高精度预测重构的冗余属性。
_Avoid_: collinearity, linear correlation

**Drift Monitoring (分布漂移监控)**:
基于统计检验判定当前因子或子系统的输入分布与回测基准区间分布是否存在显著性差异的控制机制。
_Avoid_: shift check, outlier detection

**Monotonicity (单调性)**:
因子截面分桶收益与因子数值得分排序的一致性关系。
_Avoid_: linear performance, ranking stability

## Relationships

- 一个 **Factor Card** 声明一个候选 **Factor** 的机制、假设与失效边界。
- 一个 **Subsystem Card** 声明一个包含多个输入 **Factor** 的机器学习子系统的瓶颈层与逃逸维度。
- 对候选 **Factor** 执行 **Adversarial Redundancy** 检验，判定其是否能被已有 **Factor** 库重构。
- 运行中的 **Factor** 或 **Subsystem** 必须经过 **Drift Monitoring** 来校验其数据稳定性。
- **Monotonicity** 通过分桶收益计算来检验 **Factor** 是否具备平滑的线性或非线性单调预测力。

## Example dialogue

> **研究员 (Researcher):** "我新挖出了一个成交量动量因子，回测 Sharpe 很高，能直接进入回测交易吗？"
> **风控主管 (Risk Controller):** "不行。你必须先为它提交 **Factor Card**，陈述至少 3 个失效条件。同时，该因子必须通过 **Adversarial Redundancy** 检验，证明其未被我们现有的量价因子库非线性重构。"

## Flagged ambiguities

- "feature" (特征) 与 **Factor (因子)** 的混用 — 已澄清：对于输入层及特征工程中的通用指标，统一称作特征；但对于经过治理备案、可直接产生交易决策的预测信号，统一称作 **Factor**。
- "model" (模型) 与 **Subsystem (子系统)** — 已澄清：通用的统计或机器学习模型不具备业务契约，而集成多因子并向外暴露语义信号的黑盒集成体统一称为 **Subsystem**，以其 **Subsystem Card** 作为治理边界。
