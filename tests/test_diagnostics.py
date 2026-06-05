from __future__ import annotations


import numpy as np
import polars as pl

from futures_model_factory.diagnostics.buckets import (
    long_short_by_date,
    quantile_bucket_returns,
)
from futures_model_factory.diagnostics.drift import (
    distribution_drift_summary,
    drift_gate_summary,
)
from futures_model_factory.diagnostics.ic import ic_summary, information_coefficient
from futures_model_factory.diagnostics.ic_decay import (
    ic_decay_summary,
    information_coefficient_decay,
)
from futures_model_factory.diagnostics.monotonicity import (
    monotonicity_by_date,
    monotonicity_summary,
)
from futures_model_factory.diagnostics.redundancy import (
    AdversarialRedundancyConfig,
    adversarial_redundancy_score,
    factor_redundancy_report,
)


def test_information_coefficient(mock_factor_generator) -> None:
    """验证 IC 和 Rank IC 的计算精度。"""
    # 构造高度正相关的因子与收益率
    factors, labels = mock_factor_generator(n_dates=5, assets_per_date=50, ic=0.8)
    ic_df = information_coefficient(
        factors=factors,
        labels=labels,
        factor_name="factor_test",
        label_col="next_ret",
    )

    assert ic_df.height == 5
    assert "ic" in ic_df.columns
    assert "rank_ic" in ic_df.columns
    # 期望相关系数均值处于显著正值区间
    summary = ic_summary(ic_df)
    assert summary.item(0, "mean_ic") > 0.6
    assert summary.item(0, "icir") > 0.0
    assert summary.item(0, "positive_ic_ratio") == 1.0


def test_ic_decay(mock_factor_generator) -> None:
    """验证 IC 多期衰减分析的正确性。"""
    factors, labels_single = mock_factor_generator(
        n_dates=5, assets_per_date=30, ic=0.5
    )

    # 构造多期收益率 labels (例如 lag1, lag2)
    # 使其具有人工衰减，加入独立随机噪声稀释其相关性
    lag1 = labels_single.rename({"next_ret": "ret_lag1"})
    np.random.seed(42)
    noise = np.random.normal(0, 1, lag1.height)
    lag2 = lag1.with_columns(
        (pl.col("ret_lag1") * 0.1 + pl.Series(noise) * 0.9).alias("ret_lag2")
    )
    labels = lag1.join(lag2, on=["date", "code"])

    decay_df = information_coefficient_decay(
        factors=factors,
        labels=labels,
        factor_name="factor_test",
        label_cols=["ret_lag1", "ret_lag2"],
    )

    assert decay_df.height == 2
    assert decay_df.item(0, "label_col") == "ret_lag1"
    assert decay_df.item(1, "label_col") == "ret_lag2"

    summary = ic_decay_summary(decay_df)
    assert summary.height == 1
    assert summary.item(0, "is_monotone_decay") is True
    # 验证衰减比例在预期范围内
    assert 0.1 < summary.item(0, "rank_ic_decay_ratio") < 0.8


def test_quantile_buckets_and_monotonicity(mock_factor_generator) -> None:
    """验证分桶收益率与单调性校验的逻辑。"""
    # 构造 IC = 0.95 的强单调数据集
    factors, labels = mock_factor_generator(n_dates=5, assets_per_date=60, ic=0.95)

    bucket_ret = quantile_bucket_returns(
        factors=factors,
        labels=labels,
        factor_name="factor_test",
        label_col="next_ret",
        buckets=20,
    )

    # 5 dates * 20 buckets = 100 rows
    assert bucket_ret.height == 100
    assert "bucket" in bucket_ret.columns
    assert bucket_ret.select("bucket").unique().height == 20

    # 测试 LS 多空对冲收益
    ls_ret = long_short_by_date(bucket_ret, top_bucket=20, bottom_bucket=1)
    assert ls_ret.height == 5
    assert ls_ret.select("long_short_return").mean().item(0, "long_short_return") > 0.0

    # 验证单调性校验得分
    mono_by_date = monotonicity_by_date(bucket_ret, direction="positive")
    assert mono_by_date.height == 5
    summary = monotonicity_summary(mono_by_date)

    assert summary.item(0, "mean_monotonicity_rank_corr") > 0.8
    assert summary.item(0, "mean_monotone_step_ratio") > 0.7


def test_distribution_drift(mock_drift_generator) -> None:
    """验证 PSI 和 KS 分布漂移检测的计算。"""
    # 1. 构造无漂移的数据
    ref_ok, cur_ok = mock_drift_generator(feature_cols=["f1", "f2"], drift_delta=0.0)
    drift_ok = distribution_drift_summary(ref_ok, cur_ok, feature_cols=["f1", "f2"])

    assert drift_ok.height == 2
    # 无漂移的 PSI 应较小 (通常 < 0.1)
    assert drift_ok.filter(pl.col("feature") == "f1").item(0, "psi") < 0.05
    assert drift_ok.filter(pl.col("feature") == "f1").item(0, "ks_pvalue") > 0.05

    # 对未发生漂移的情况，我们将 ks_threshold 提高到 0.15 避开采样波动的误伤
    decision_ok = drift_gate_summary(drift_ok, psi_threshold=0.1, ks_threshold=0.15)
    assert decision_ok.filter(pl.col("feature") == "f1").item(0, "breach") is False

    # 2. 构造有明显分布偏移的数据 (漂移量设为 1.5 倍标准差)
    ref_drift, cur_drift = mock_drift_generator(feature_cols=["f1"], drift_delta=1.5)
    drift_bad = distribution_drift_summary(ref_drift, cur_drift, feature_cols=["f1"])

    # PSI 显著变大
    assert drift_bad.item(0, "psi") > 0.5
    # KS 显著性水平 pvalue 逼近 0 拒绝原假设 (说明存在漂移，所以 ks_stat 应当非常大)
    assert drift_bad.item(0, "ks_stat") > 0.3

    decision_bad = drift_gate_summary(drift_bad, psi_threshold=0.15, ks_threshold=0.1)
    assert decision_bad.item(0, "breach") is True


def test_linear_and_adversarial_redundancy(
    mock_factor_generator, mock_library_generator
) -> None:
    """验证线性相关系数过滤与非线性对抗重构检验。"""
    n_dates = 8
    assets = 50

    # 构造因子库 (包含 existing_f1, existing_f2)
    library = mock_library_generator(
        n_dates=n_dates,
        assets_per_date=assets,
        factor_names=["existing_f1", "existing_f2"],
    )

    # 1. 构造一个与已有库完全不相关的候选因子
    # 直接由 mock_factor_generator 独立生成 (因为 seed 不同，统计上独立)
    candidate_independent, _ = mock_factor_generator(
        n_dates=n_dates, assets_per_date=assets, seed=100
    )

    # 线性校验
    rep_indep = factor_redundancy_report(
        candidate=candidate_independent,
        library=library,
        factor_name="independent_f",
        category="price_volume",
    )
    assert rep_indep.item(0, "is_linearly_redundant") is False

    # 对抗校验 (非线性)
    cfg = AdversarialRedundancyConfig(category="price_volume", train_fraction=0.7)
    adv_indep = adversarial_redundancy_score(
        candidate=candidate_independent,
        library=library,
        factor_name="independent_f",
        config=cfg,
    )
    assert adv_indep.item(0, "is_adversarially_redundant") is False

    # 2. 构造一个可以被已有因子库高度重构的冗余因子
    # Y = 1.5 * existing_f1 - 0.8 * existing_f2^2 + noise
    # 注意此时 library 已经是 wide 格式，直接取其特征列
    redundant_series = (
        1.5 * library.get_column("existing_f1")
        - 0.8 * (library.get_column("existing_f2") ** 2)
        + pl.Series(np.random.normal(0, 0.1, library.height))
    )
    candidate_redundant = pl.DataFrame(
        {
            "date": library.get_column("date"),
            "code": library.get_column("code"),
            "factor_value": redundant_series,
        }
    )

    # 对其运行对抗检验，其 R2 应该大于 threshold (price_volume 为 0.4)
    adv_red = adversarial_redundancy_score(
        candidate=candidate_redundant,
        library=library,
        factor_name="redundant_f",
        config=cfg,
    )
    assert adv_red.item(0, "adversarial_r2") > 0.6
    assert adv_red.item(0, "is_adversarially_redundant") is True
