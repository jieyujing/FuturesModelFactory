from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import polars as pl

from futures_model_factory.diagnostics.buckets import quantile_bucket_returns
from futures_model_factory.diagnostics.drift import (
    distribution_drift_summary,
    drift_gate_summary,
)
from futures_model_factory.diagnostics.ic import ic_summary, information_coefficient
from futures_model_factory.diagnostics.monotonicity import (
    monotonicity_by_date,
    monotonicity_summary,
)
from futures_model_factory.diagnostics.redundancy import (
    AdversarialRedundancyConfig,
    FactorCategory,
    adversarial_redundancy_score,
    factor_redundancy_report,
)
from futures_model_factory.registry.factor_cards import FactorCard, load_factor_card
from futures_model_factory.utils.provenance import RunProvenance, make_run_provenance

EvaluationStatus = Literal["APPROVED", "REJECTED_REDUNDANT", "REJECTED_DRIFTED"]


@dataclass(frozen=True)
class FactorEvaluationReport:
    """因子端到端评估准入报告。"""

    factor_name: str
    status: EvaluationStatus
    provenance: RunProvenance | None
    redundancy_report: pl.DataFrame
    adversarial_redundancy: pl.DataFrame
    ic_summary: pl.DataFrame
    monotonicity_summary: pl.DataFrame
    drift_summary: pl.DataFrame
    drift_decision: pl.DataFrame
    details: dict[str, Any]


class FactorEvaluator:
    """因子评估器，用于运行契约化准入校验。

    串联卡片读取、线性与非线性对抗冗余校验、IC/分桶单调性诊断以及输入数据分布漂移监控。
    """

    def __init__(
        self,
        *,
        category: FactorCategory = "price_volume",
        psi_threshold: float = 0.25,
        ks_threshold: float = 0.05,
        adversarial_config: AdversarialRedundancyConfig | None = None,
    ) -> None:
        """初始化评估器参数。

        Args:
            category: 因子类别，决定冗余度校验的阈值。
            psi_threshold: PSI 分布漂移警示阈值，默认 0.25 表示显著漂移。
            ks_threshold: KS 检验漂移判断阈值（最大统计距离），默认 0.05。
            adversarial_config: 对抗冗余 GBDT 模型配置。
        """
        self.category = category
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.adversarial_config = adversarial_config or AdversarialRedundancyConfig(
            category=category
        )

    def evaluate(
        self,
        card: FactorCard | str | Path,
        *,
        candidate: pl.DataFrame,
        library: pl.DataFrame,
        labels: pl.DataFrame,
        label_col: str,
        drift_reference: pl.DataFrame,
        drift_current: pl.DataFrame,
        config_path_for_provenance: str | Path = "config.yaml",
        data_paths_for_provenance: list[str | Path] | None = None,
    ) -> FactorEvaluationReport:
        """执行因子的完整性校验与统计准入测试。

        Args:
            card: 因子卡片对象或其 YAML 配置文件路径。
            candidate: 候选因子数据 (必含 date, code, factor_value)。
            library: 已存在因子库数据。
            labels: 收益率标签数据 (必含 date, code, label_col)。
            label_col: 收益率标签列名。
            drift_reference: 分布漂移基准参考区间数据集。
            drift_current: 分布漂移当前校验区间数据集。
            config_path_for_provenance: 配置文件路径，用于物源跟踪。
            data_paths_for_provenance: 数据文件路径列表，用于物源跟踪。

        Returns:
            FactorEvaluationReport: 准入判定报告。
        """
        # 1. 加载和基本校验因子卡片 (如果传入的是路径则进行加载)
        if isinstance(card, (str, Path)):
            resolved_card = load_factor_card(card)
        else:
            resolved_card = card

        factor_name = resolved_card.name

        # 2. 冗余性检验 (线性与非线性)
        red_report = factor_redundancy_report(
            candidate=candidate,
            library=library,
            factor_name=factor_name,
            category=self.category,
        )
        adv_red_report = adversarial_redundancy_score(
            candidate=candidate,
            library=library,
            factor_name=factor_name,
            config=self.adversarial_config,
        )

        is_lin_redundant = red_report.item(0, "is_linearly_redundant")
        is_adv_redundant = adv_red_report.item(0, "is_adversarially_redundant")

        # 默认为 False，只有明确为 True 时才判定为冗余
        is_redundant = bool(is_lin_redundant) or bool(is_adv_redundant)

        # 3. 诊断计算 (IC & Monotonicity)
        ic_by_date = information_coefficient(
            factors=candidate,
            labels=labels,
            factor_name=factor_name,
            label_col=label_col,
        )
        ic_sum = ic_summary(ic_by_date)

        # 默认计算 20 个分桶
        bucket_ret = quantile_bucket_returns(
            factors=candidate,
            labels=labels,
            factor_name=factor_name,
            label_col=label_col,
            buckets=20,
        )
        # 根据预期符号方向判定单调性
        expected_direction = (
            "positive" if resolved_card.expected_sign == "1" else "negative"
        )
        mono_by_date = monotonicity_by_date(bucket_ret, direction=expected_direction)
        mono_sum = monotonicity_summary(mono_by_date)

        # 4. 数据漂移监控
        # 提取因子卡片中定义的所有 inputs 进行漂移监测
        drift_sum = distribution_drift_summary(
            reference=drift_reference,
            current=drift_current,
            feature_cols=resolved_card.inputs,
        )
        drift_dec = drift_gate_summary(
            drift_sum,
            psi_threshold=self.psi_threshold,
            ks_threshold=self.ks_threshold,
        )

        is_drifted = drift_dec.get_column("breach").any()

        # 5. 准入状态决策
        if is_redundant:
            status: EvaluationStatus = "REJECTED_REDUNDANT"
        elif is_drifted:
            status = "REJECTED_DRIFTED"
        else:
            status = "APPROVED"

        # 6. 绑定 RunProvenance (仅在通过准入时生成)
        provenance: RunProvenance | None = None
        if status == "APPROVED":
            paths = data_paths_for_provenance or []
            provenance = make_run_provenance(config_path_for_provenance, paths)

        details = {
            "is_linearly_redundant": is_lin_redundant,
            "is_adversarially_redundant": is_adv_redundant,
            "max_abs_linear_correlation": red_report.item(0, "max_abs_correlation"),
            "adversarial_r2": adv_red_report.item(0, "adversarial_r2"),
            "drift_breached_features": drift_dec.filter(pl.col("breach"))
            .get_column("feature")
            .to_list(),
        }

        return FactorEvaluationReport(
            factor_name=factor_name,
            status=status,
            provenance=provenance,
            redundancy_report=red_report,
            adversarial_redundancy=adv_red_report,
            ic_summary=ic_sum,
            monotonicity_summary=mono_sum,
            drift_summary=drift_sum,
            drift_decision=drift_dec,
            details=details,
        )
