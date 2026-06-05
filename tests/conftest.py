from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import polars as pl
import pytest


@pytest.fixture
def temp_registry() -> Generator[Path, None, None]:
    """生成带有合规和不合规卡片文件的临时目录。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        factors_dir = path / "factors"
        subsystems_dir = path / "subsystems"
        factors_dir.mkdir()
        subsystems_dir.mkdir()

        # 1. 写入一个完全合规的因子卡片
        factor_ok = """
name: "factor_volume_momentum"
status: "candidate"
hypothesis: "在高成交量伴随价格突破时，资金动量因子具有正向预测力。"
mechanism_type: "trend_following"
expected_sign: "1"
inputs:
  - "close_price"
  - "volume"
failure_conditions:
  - { condition: "market_sideways", metric: "atr_ratio_below_0_1" }
  - { condition: "low_liquidity", metric: "daily_volume_below_1000" }
  - { condition: "high_correlation_with_index", metric: "beta_to_index_above_1_2" }
kill_conditions:
  - "drawdown_above_15pct"
search_context:
  optimizer: "none"
  sample_period: "2020-01-01 to 2023-12-31"
deep_attribution:
  market_structure_parent: "volume_climax"
log_contrast_proxy:
  reference_factor: "volume_ma20"
"""
        # 注意：对于因子卡片中的 failure_conditions.metric，因为 factor_cards.py 不做 _reject_placeholders 检查，
        # 所以即使这里含有 < 或 >，也不会触发类似 subsystem_cards.py 里的占位符报错。
        # 但为了避免未来的隐患，我们将 atr_ratio < 0.1 改为 atr_ratio_below_0_1。
        # daily_volume < 1000 改为 daily_volume_below_1000 以彻底规避 placeholder 检测误判。
        (factors_dir / "factor_volume_momentum.yaml").write_text(factor_ok, encoding="utf-8")

        # 2. 写入一个不合规的因子卡片 (缺失必填项，如 deep_attribution)
        factor_bad = """
name: "factor_bad"
status: "candidate"
hypothesis: "测试缺失字段"
mechanism_type: "trend_following"
expected_sign: "1"
inputs:
  - "close_price"
failure_conditions:
  - { condition: "market_sideways", metric: "atr_ratio_below_0_1" }
kill_conditions:
  - "drawdown_above_15pct"
search_context:
  optimizer: "none"
"""
        (factors_dir / "factor_bad.yaml").write_text(factor_bad, encoding="utf-8")

        # 3. 写入一个包含 TBD 占位符的子系统卡片
        subsystem_placeholder = """
subsystem_id: "subsystem_placeholder"
input_granularity: "raw_bars"
bottleneck_location: "input_layer"
escape_dimensions:
  - "scalar_projection"
interpretability_layers_kept:
  - "L1"
  - "L2"
interpretability_layers_given_up:
  - "L3"
mechanism_class: "TBD"
profit_source_category: "statistical_arbitrage"
expected_mechanism_sign: "1"
scope_conditions:
  active_when: ["TBD"]
  inactive_when: ["TBD"]
baseline_to_beat:
  description: "TBD"
  metric: "incremental_IR"
  threshold: 0.5
  evaluation_window: "1y"
kill_condition:
  metric: "drawdown"
  threshold: 0.15
  consecutive_days: 5
  rearm_rule: "manual"
drift_monitoring:
  input_distribution_check: "PSI"
  threshold: 0.25
  action_on_breach: "deactivate"
search_context:
  architectures_considered: 10
  encoders_considered: "lstm"
  horizons_considered: ["5d"]
  selection_rule: "max_ir"
  research_ledger_entries: ["entry1"]
compensations:
  for_L3_readability_surrender: "use_attention_weights"
output_signals:
  - name: "signal_1"
    range: "[-1, 1]"
    semantics: "trend"
audit:
  lifecycle_stage: "draft"
"""
        (subsystems_dir / "subsystem_bad.yaml").write_text(
            subsystem_placeholder, encoding="utf-8"
        )

        # 4. 写入一个完全合规的子系统卡片
        subsystem_ok = """
subsystem_id: "subsystem_ok"
input_granularity: "raw_bars"
bottleneck_location: "input_layer"
escape_dimensions:
  - "scalar_projection"
interpretability_layers_kept:
  - "L1"
  - "L2"
interpretability_layers_given_up:
  - "L3"
mechanism_class: "trend"
profit_source_category: "statistical_arbitrage"
expected_mechanism_sign: "1"
scope_conditions:
  active_when: ["atr_ratio_above_0_1"]
  inactive_when: ["atr_ratio_below_0_1"]
baseline_to_beat:
  description: "beat simple trend following"
  metric: "incremental_IR"
  threshold: 0.5
  evaluation_window: "1y"
kill_condition:
  metric: "drawdown"
  threshold: 0.15
  consecutive_days: 5
  rearm_rule: "manual"
drift_monitoring:
  input_distribution_check: "PSI"
  threshold: 0.25
  action_on_breach: "deactivate"
search_context:
  architectures_considered: 10
  encoders_considered: "lstm"
  horizons_considered: ["5d"]
  selection_rule: "max_ir"
  research_ledger_entries: ["entry1"]
compensations:
  for_L3_readability_surrender: "use_attention_weights"
output_signals:
  - name: "signal_1"
    range: "[-1, 1]"
    semantics: "trend"
audit:
  lifecycle_stage: "draft"
"""
        (subsystems_dir / "subsystem_ok.yaml").write_text(subsystem_ok, encoding="utf-8")

        yield path


@pytest.fixture
def mock_factor_generator():
    """生成具有可控 IC 和单调分桶收益特征数据的发生器。"""

    def _generate(
        *,
        n_dates: int = 10,
        assets_per_date: int = 40,
        ic: float = 0.1,
        seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """生成因子 DataFrame 和收益率 DataFrame。

        Y = ic * X + sqrt(1 - ic^2) * Z
        """
        np.random.seed(seed)
        dates = [f"2026-06-{i + 1:02d}" for i in range(n_dates)]
        codes = [f"FUT{j + 1:03d}" for j in range(assets_per_date)]

        factors_list = []
        labels_list = []

        for date in dates:
            # 因子值服从标准正态分布
            x = np.random.normal(0, 1, assets_per_date)
            # 随机噪声
            z = np.random.normal(0, 1, assets_per_date)
            # 收益率与因子值具备指定的相关性
            y = ic * x + np.sqrt(1 - ic**2) * z

            for code, x_val, y_val in zip(codes, x, y):
                factors_list.append(
                    {"date": date, "code": code, "factor_value": float(x_val)}
                )
                labels_list.append(
                    {"date": date, "code": code, "next_ret": float(y_val)}
                )

        return pl.DataFrame(factors_list), pl.DataFrame(labels_list)

    return _generate


@pytest.fixture
def mock_drift_generator():
    """分布漂移数据发生器，返回 reference 和 current 数据集。"""

    def _generate(
        *,
        feature_cols: list[str],
        size_ref: int = 1000,
        size_cur: int = 500,
        drift_delta: float = 0.0,  # 偏移量，0.0 表示不发生漂移
        seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        np.random.seed(seed)
        ref_data = {}
        cur_data = {}

        for col in feature_cols:
            ref_data[col] = np.random.normal(0.0, 1.0, size_ref)
            cur_data[col] = np.random.normal(drift_delta, 1.0, size_cur)

        return pl.DataFrame(ref_data), pl.DataFrame(cur_data)

    return _generate


@pytest.fixture
def mock_library_generator():
    """生成包含多因子已有库的发生器。"""

    def _generate(
        *,
        n_dates: int = 10,
        assets_per_date: int = 40,
        factor_names: list[str],
        seed: int = 42,
    ) -> pl.DataFrame:
        np.random.seed(seed)
        dates = [f"2026-06-{i + 1:02d}" for i in range(n_dates)]
        codes = [f"FUT{j + 1:03d}" for j in range(assets_per_date)]

        records = []
        for date in dates:
            for code in codes:
                row = {"date": date, "code": code}
                for factor in factor_names:
                    row[factor] = float(np.random.normal(0, 1))
                records.append(row)

        return pl.DataFrame(records)

    return _generate
