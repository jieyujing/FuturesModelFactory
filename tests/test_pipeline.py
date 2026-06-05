from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl

from futures_model_factory.pipeline import FactorEvaluator
from futures_model_factory.registry.factor_cards import load_factor_card


def test_pipeline_approved_flow(
    temp_registry: Path,
    mock_factor_generator,
    mock_library_generator,
    mock_drift_generator,
) -> None:
    """验证一个完全合规、表现良好、且没有冗余和漂移的因子，可以被 APPROVED 准入，并绑定 Provenance。"""
    card_path = temp_registry / "factors" / "factor_volume_momentum.yaml"
    card = load_factor_card(card_path)

    # 1. 模拟生成因子与 labels 数据集
    candidate, labels = mock_factor_generator(
        n_dates=10, assets_per_date=50, ic=0.1, seed=42
    )

    # 2. 模拟生成完全独立的因子库，不会产生冗余
    library = mock_library_generator(
        n_dates=10, assets_per_date=50, factor_names=["existing_f1"], seed=100
    )

    # 3. 模拟漂移监控数据 (输入特征 close_price, volume)，设偏移为 0.0 表示完全无漂移
    drift_ref, drift_cur = mock_drift_generator(
        feature_cols=["close_price", "volume"], drift_delta=0.0, seed=42
    )

    # 4. 创建临时配置文件，用于 Provenance 物源追踪
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_config:
        config_path = tmp_config.name
        tmp_config.write(b"optimizer: none\nsample_period: 2026")

    # 创建一个临时数据文件，用于 Provenance 哈希
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_data:
        data_path = tmp_data.name
        tmp_data.write(b"date,code,factor_value\n2026-06-01,FUT001,0.5")

    try:
        evaluator = FactorEvaluator(
            category="price_volume",
            psi_threshold=0.25,
            ks_threshold=0.15,
        )

        report = evaluator.evaluate(
            card=card,
            candidate=candidate,
            library=library,
            labels=labels,
            label_col="next_ret",
            drift_reference=drift_ref,
            drift_current=drift_cur,
            config_path_for_provenance=config_path,
            data_paths_for_provenance=[data_path],
        )

        # 验证报告内容
        assert report.factor_name == "factor_volume_momentum"
        assert report.status == "APPROVED"
        assert report.provenance is not None
        assert report.provenance.config_path == config_path
        # 详细字典中记录的冗余校验和漂移判定
        assert report.details["is_linearly_redundant"] is False
        assert report.details["is_adversarially_redundant"] is False
        assert len(report.details["drift_breached_features"]) == 0
        assert report.ic_summary.height == 1
        assert report.monotonicity_summary.height == 1

    finally:
        # 清理临时文件
        Path(config_path).unlink()
        Path(data_path).unlink()


def test_pipeline_rejected_redundant(
    temp_registry: Path,
    mock_factor_generator,
    mock_library_generator,
    mock_drift_generator,
) -> None:
    """验证当因子与已有因子库高度相关 (冗余) 时，会被判定为 REJECTED_REDUNDANT 并拒绝入库，不产生 Provenance。"""
    card_path = temp_registry / "factors" / "factor_volume_momentum.yaml"
    card = load_factor_card(card_path)

    # 1. 模拟生成因子与 labels 数据集
    candidate, labels = mock_factor_generator(
        n_dates=8, assets_per_date=50, ic=0.1, seed=42
    )

    # 2. 模拟生成与候选因子完全一致的因子库 (以 candidate 因子作为 library 的其中一列)
    # 这样它们线性相关性为 1.0，必然会被冗余过滤器拦截
    library_wide = candidate.select(
        "date", "code", pl.col("factor_value").alias("factor_volume_momentum")
    )
    # 把格式转为 long 形式以配合 _library_to_wide
    library = library_wide.unpivot(
        variable_name="factor", value_name="factor_value", index=["date", "code"]
    )

    # 3. 模拟漂移监控数据 (无漂移)
    drift_ref, drift_cur = mock_drift_generator(
        feature_cols=["close_price", "volume"], drift_delta=0.0, seed=42
    )

    evaluator = FactorEvaluator(category="price_volume")
    report = evaluator.evaluate(
        card=card,
        candidate=candidate,
        library=library,
        labels=labels,
        label_col="next_ret",
        drift_reference=drift_ref,
        drift_current=drift_cur,
    )

    assert report.status == "REJECTED_REDUNDANT"
    assert report.provenance is None
    assert report.details["is_linearly_redundant"] is True


def test_pipeline_rejected_drifted(
    temp_registry: Path,
    mock_factor_generator,
    mock_library_generator,
    mock_drift_generator,
) -> None:
    """验证当输入特征发生显著性分布漂移时，会被判定为 REJECTED_DRIFTED 并拒绝入库。"""
    card_path = temp_registry / "factors" / "factor_volume_momentum.yaml"
    card = load_factor_card(card_path)

    # 1. 模拟生成因子与 labels 数据集 (无冗余)
    candidate, labels = mock_factor_generator(
        n_dates=8, assets_per_date=50, ic=0.1, seed=42
    )
    library = mock_library_generator(
        n_dates=8, assets_per_date=50, factor_names=["existing_f1"], seed=100
    )

    # 2. 模拟漂移监控数据，设置 drift_delta 为 2.0 (会引发强烈的 PSI 和 KS 漂移拦截)
    drift_ref, drift_cur = mock_drift_generator(
        feature_cols=["close_price", "volume"], drift_delta=2.0, seed=42
    )

    evaluator = FactorEvaluator(
        category="price_volume", psi_threshold=0.25, ks_threshold=0.05
    )
    report = evaluator.evaluate(
        card=card,
        candidate=candidate,
        library=library,
        labels=labels,
        label_col="next_ret",
        drift_reference=drift_ref,
        drift_current=drift_cur,
    )

    assert report.status == "REJECTED_DRIFTED"
    assert report.provenance is None
    assert "close_price" in report.details["drift_breached_features"]
    assert "volume" in report.details["drift_breached_features"]
