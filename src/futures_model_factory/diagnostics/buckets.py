from __future__ import annotations

import polars as pl


def quantile_bucket_returns(
    factors: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    factor_name: str,
    label_col: str,
    buckets: int = 20,
) -> pl.DataFrame:
    """Compute per-date factor bucket returns with bucket 1 as the lowest scores.

    Formal factor diagnostics use at least 20 buckets to expose weak
    monotonicity and hidden tail dependence.
    """
    if buckets < 20:
        raise ValueError("buckets must be at least 20 for factor bucket diagnostics")

    joined = (
        factors.join(labels, on=["date", "code"], how="inner")
        .filter(pl.col("factor_value").is_not_null() & pl.col(label_col).is_not_null())
        .with_columns(
            pl.col("factor_value").rank("ordinal").over("date").alias("rank_in_date"),
            pl.len().over("date").alias("date_count"),
        )
        .with_columns(
            (((pl.col("rank_in_date") - 1) * buckets / pl.col("date_count")).floor() + 1)
            .clip(1, buckets)
            .cast(pl.Int64)
            .alias("bucket")
        )
    )

    return (
        joined.group_by("date", "bucket")
        .agg(
            pl.lit(factor_name).alias("factor"),
            pl.col(label_col).mean().alias("mean_forward_return"),
            pl.len().alias("n"),
        )
        .select("date", "factor", "bucket", "mean_forward_return", "n")
        .sort(["date", "bucket"])
    )


def long_short_by_date(bucket_returns: pl.DataFrame, *, top_bucket: int = 20, bottom_bucket: int = 1) -> pl.DataFrame:
    """Return top-minus-bottom bucket spread by date."""
    wide = bucket_returns.filter(pl.col("bucket").is_in([bottom_bucket, top_bucket])).pivot(
        values="mean_forward_return", index="date", on="bucket"
    )
    top = str(top_bucket)
    bottom = str(bottom_bucket)
    return wide.with_columns((pl.col(top) - pl.col(bottom)).alias("long_short_return")).select(
        "date", "long_short_return"
    )
