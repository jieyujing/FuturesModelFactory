from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor  # type: ignore[import-untyped]
from sklearn.metrics import r2_score  # type: ignore[import-untyped]

FactorCategory = Literal["price_volume", "fundamental", "alternative"]

_REDUNDANCY_THRESHOLDS: dict[FactorCategory, float] = {
    "price_volume": 0.4,
    "fundamental": 0.6,
    "alternative": 0.7,
}


@dataclass(frozen=True)
class AdversarialRedundancyConfig:
    """Settings for predicting a candidate factor from the existing library."""

    category: FactorCategory = "price_volume"
    train_fraction: float = 0.7
    min_train_rows: int = 20
    min_test_rows: int = 5
    random_state: int = 42


def _library_to_wide(library: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    if {"date", "code", "factor", "factor_value"}.issubset(library.columns):
        wide = library.pivot(values="factor_value", index=["date", "code"], on="factor")
        feature_cols = [col for col in wide.columns if col not in {"date", "code"}]
        return wide, feature_cols

    if {"date", "code"}.issubset(library.columns):
        feature_cols = [col for col in library.columns if col not in {"date", "code"}]
        return library, feature_cols

    raise ValueError("library must be wide with date/code plus factor columns or long with factor/factor_value")


def _joined_candidate_library(candidate: pl.DataFrame, library: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    required = {"date", "code", "factor_value"}
    if not required.issubset(candidate.columns):
        raise ValueError("candidate must contain date, code, factor_value")

    wide, feature_cols = _library_to_wide(library)
    if not feature_cols:
        raise ValueError("library must contain at least one factor column")

    joined = (
        candidate.select("date", "code", pl.col("factor_value").alias("candidate_factor_value"))
        .join(wide, on=["date", "code"], how="inner")
        .sort(["date", "code"])
    )
    return joined, feature_cols


def factor_redundancy_report(
    candidate: pl.DataFrame,
    library: pl.DataFrame,
    *,
    factor_name: str,
    category: FactorCategory = "price_volume",
) -> pl.DataFrame:
    """Measure same-date linear redundancy against an existing factor library.

    The output is intentionally one row per candidate: it is a gate, not a model.
    Use adversarial_redundancy_score for nonlinear predictability.
    """
    joined, feature_cols = _joined_candidate_library(candidate, library)
    corr_exprs = [pl.corr("candidate_factor_value", col).alias(col) for col in feature_cols]
    corr_long = (
        joined.select(corr_exprs)
        .unpivot(variable_name="existing_factor", value_name="correlation")
        .with_columns(pl.col("correlation").abs().alias("abs_correlation"))
        .filter(pl.col("correlation").is_not_null())
    )

    if corr_long.is_empty():
        return pl.DataFrame({
            "factor": [factor_name],
            "n": [joined.height],
            "category": [category],
            "redundancy_threshold": [_REDUNDANCY_THRESHOLDS[category]],
            "max_abs_correlation": [None],
            "mean_abs_correlation": [None],
            "most_correlated_factor": [None],
            "is_linearly_redundant": [None],
        })

    top = corr_long.sort("abs_correlation", descending=True).head(1)
    summary = corr_long.select(
        pl.col("abs_correlation").max().alias("max_abs_correlation"),
        pl.col("abs_correlation").mean().alias("mean_abs_correlation"),
    )
    max_corr = summary.item(0, "max_abs_correlation")
    threshold = _REDUNDANCY_THRESHOLDS[category]

    return pl.DataFrame({
        "factor": [factor_name],
        "n": [joined.height],
        "category": [category],
        "redundancy_threshold": [threshold],
        "max_abs_correlation": [max_corr],
        "mean_abs_correlation": [summary.item(0, "mean_abs_correlation")],
        "most_correlated_factor": [top.item(0, "existing_factor")],
        "is_linearly_redundant": [bool(max_corr is not None and max_corr >= threshold)],
    })


def adversarial_redundancy_score(
    candidate: pl.DataFrame,
    library: pl.DataFrame,
    *,
    factor_name: str,
    config: AdversarialRedundancyConfig | None = None,
) -> pl.DataFrame:
    """Predict a candidate factor from existing factors using a nonlinear model.

    The date-sorted holdout split keeps this as a redundancy gate instead of a
    same-sample fit. A high R2 means the new factor is largely reconstructable
    from existing library signals and should be treated as redundant until proven
    otherwise.
    """
    cfg = config or AdversarialRedundancyConfig()
    if not 0.0 < cfg.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    joined, feature_cols = _joined_candidate_library(candidate, library)
    clean = joined.filter(pl.col("candidate_factor_value").is_not_null())
    if clean.height < cfg.min_train_rows + cfg.min_test_rows:
        return pl.DataFrame({
            "factor": [factor_name],
            "n_train": [0],
            "n_test": [0],
            "category": [cfg.category],
            "redundancy_threshold": [_REDUNDANCY_THRESHOLDS[cfg.category]],
            "adversarial_r2": [None],
            "is_adversarially_redundant": [None],
        })

    dates = clean.select("date").unique().sort("date").get_column("date").to_list()
    split_index = max(1, min(len(dates) - 1, int(len(dates) * cfg.train_fraction)))
    train_dates = set(dates[:split_index])
    train = clean.filter(pl.col("date").is_in(train_dates))
    test = clean.filter(~pl.col("date").is_in(train_dates))

    if train.height < cfg.min_train_rows or test.height < cfg.min_test_rows:
        return pl.DataFrame({
            "factor": [factor_name],
            "n_train": [train.height],
            "n_test": [test.height],
            "category": [cfg.category],
            "redundancy_threshold": [_REDUNDANCY_THRESHOLDS[cfg.category]],
            "adversarial_r2": [None],
            "is_adversarially_redundant": [None],
        })

    medians = {col: train.get_column(col).median() for col in feature_cols}
    fill_exprs = [pl.col(col).fill_null(medians[col] if medians[col] is not None else 0.0).alias(col) for col in feature_cols]
    train = train.with_columns(fill_exprs)
    test = test.with_columns(fill_exprs)

    x_train = train.select(feature_cols).to_numpy()
    y_train = train.get_column("candidate_factor_value").to_numpy()
    x_test = test.select(feature_cols).to_numpy()
    y_test = test.get_column("candidate_factor_value").to_numpy()

    model = GradientBoostingRegressor(random_state=cfg.random_state, max_depth=2, n_estimators=80, learning_rate=0.05)
    model.fit(x_train, y_train)
    raw_score = float(r2_score(y_test, model.predict(x_test)))
    score: float | None = None if np.isnan(raw_score) else raw_score
    threshold = _REDUNDANCY_THRESHOLDS[cfg.category]

    return pl.DataFrame({
        "factor": [factor_name],
        "n_train": [train.height],
        "n_test": [test.height],
        "category": [cfg.category],
        "redundancy_threshold": [threshold],
        "adversarial_r2": [score],
        "is_adversarially_redundant": [bool(score is not None and score >= threshold)],
    })
