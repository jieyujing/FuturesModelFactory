from __future__ import annotations

from typing import Literal

import polars as pl
from scipy.stats import spearmanr  # type: ignore[import-untyped]

Direction = Literal["positive", "negative"]


def monotonicity_by_date(
    bucket_returns: pl.DataFrame, *, direction: Direction = "positive"
) -> pl.DataFrame:
    """Score whether bucket returns move monotonically with factor rank by date."""
    required = {"date", "factor", "bucket", "mean_forward_return"}
    if not required.issubset(bucket_returns.columns):
        raise ValueError(
            "bucket_returns must contain date, factor, bucket, mean_forward_return"
        )
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be positive or negative")

    rows = []
    sorted_buckets = bucket_returns.sort(["factor", "date", "bucket"])
    for keys, frame in sorted_buckets.group_by(["factor", "date"], maintain_order=True):
        factor, date = keys
        buckets = frame.get_column("bucket").to_list()
        returns = frame.get_column("mean_forward_return").to_list()
        if len(buckets) < 2:
            rho = None
            violation_count = None
            monotone_ratio = None
            top_bottom_spread = None
        else:
            if len(set(returns)) <= 1 or len(set(buckets)) <= 1:
                rho = None
            else:
                rho_value = spearmanr(buckets, returns).statistic
                rho = None if rho_value != rho_value else float(rho_value)
            diffs = [right - left for left, right in zip(returns, returns[1:])]
            if direction == "positive":
                violations = sum(diff < 0 for diff in diffs)
                top_bottom_spread = returns[-1] - returns[0]
            else:
                violations = sum(diff > 0 for diff in diffs)
                top_bottom_spread = returns[0] - returns[-1]
                rho = None if rho is None else -rho
            violation_count = int(violations)
            monotone_ratio = 1.0 - violation_count / len(diffs)
        rows.append(
            {
                "date": date,
                "factor": factor,
                "direction": direction,
                "bucket_count": len(buckets),
                "monotonicity_rank_corr": rho,
                "violation_count": violation_count,
                "monotone_step_ratio": monotone_ratio,
                "top_bottom_spread": top_bottom_spread,
            }
        )

    return pl.DataFrame(rows).sort(["factor", "date"])


def monotonicity_summary(monotonicity: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-date monotonicity without hiding violations."""
    return monotonicity.group_by("factor").agg(
        pl.col("direction").first().alias("direction"),
        pl.col("monotonicity_rank_corr").mean().alias("mean_monotonicity_rank_corr"),
        pl.col("monotone_step_ratio").mean().alias("mean_monotone_step_ratio"),
        (pl.col("violation_count") == 0).mean().alias("perfect_monotonicity_ratio"),
        pl.col("top_bottom_spread").mean().alias("mean_top_bottom_spread"),
        pl.len().alias("periods"),
    )
