from __future__ import annotations

import polars as pl


def information_coefficient(
    factors: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    factor_name: str,
    label_col: str,
) -> pl.DataFrame:
    """Compute per-date Pearson IC and rank IC for one factor."""
    joined = (
        factors.join(labels, on=["date", "code"], how="inner")
        .filter(pl.col("factor_value").is_not_null() & pl.col(label_col).is_not_null())
        .with_columns(
            pl.col("factor_value").rank("average").over("date").alias("factor_rank"),
            pl.col(label_col).rank("average").over("date").alias("label_rank"),
        )
    )

    return (
        joined.group_by("date")
        .agg(
            pl.lit(factor_name).alias("factor"),
            pl.corr("factor_value", label_col).alias("ic"),
            pl.corr("factor_rank", "label_rank").alias("rank_ic"),
            pl.len().alias("n"),
        )
        .sort("date")
    )


def ic_summary(ic_by_date: pl.DataFrame) -> pl.DataFrame:
    """Summarize IC stability without hiding the per-date evidence."""
    return ic_by_date.group_by("factor").agg(
        pl.col("ic").mean().alias("mean_ic"),
        pl.col("ic").std().alias("std_ic"),
        (pl.col("ic").mean() / pl.col("ic").std()).alias("icir"),
        (pl.col("ic") > 0).mean().alias("positive_ic_ratio"),
        pl.col("rank_ic").mean().alias("mean_rank_ic"),
        pl.col("n").mean().alias("mean_n"),
    )
