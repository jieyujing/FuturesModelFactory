from __future__ import annotations

import polars as pl

from futures_model_factory.diagnostics.ic import information_coefficient


def information_coefficient_decay(
    factors: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    factor_name: str,
    label_cols: list[str],
) -> pl.DataFrame:
    """Compute IC summaries across multiple forward-return horizons."""
    if not label_cols:
        raise ValueError("label_cols must contain at least one label column")

    rows: list[pl.DataFrame] = []
    for horizon_index, label_col in enumerate(label_cols, start=1):
        ic = information_coefficient(
            factors, labels, factor_name=factor_name, label_col=label_col
        )
        rows.append(
            ic.group_by("factor").agg(
                pl.lit(label_col).alias("label_col"),
                pl.lit(horizon_index).alias("horizon_order"),
                pl.col("ic").mean().alias("mean_ic"),
                pl.col("rank_ic").mean().alias("mean_rank_ic"),
                pl.col("ic").std().alias("std_ic"),
                (pl.col("ic").mean() / pl.col("ic").std()).alias("icir"),
                (pl.col("ic") > 0).mean().alias("positive_ic_ratio"),
                pl.len().alias("periods"),
                pl.col("n").mean().alias("mean_n"),
            )
        )

    return pl.concat(rows).sort("horizon_order")


def ic_decay_summary(ic_decay: pl.DataFrame) -> pl.DataFrame:
    """Summarize whether IC decays as the forecast horizon extends."""
    required = {"factor", "horizon_order", "mean_rank_ic"}
    if not required.issubset(ic_decay.columns):
        raise ValueError("ic_decay must contain factor, horizon_order, mean_rank_ic")

    by_factor = []
    for factor, frame in ic_decay.sort("horizon_order").group_by(
        "factor", maintain_order=True
    ):
        values = frame.get_column("mean_rank_ic").to_list()
        abs_values = [abs(v) for v in values if v is not None]
        if len(abs_values) < 2:
            decay_ratio = None
            monotone_decay = None
        else:
            first = abs_values[0]
            last = abs_values[-1]
            decay_ratio = None if first == 0 else last / first
            monotone_decay = all(
                left >= right for left, right in zip(abs_values, abs_values[1:])
            )
        factor_name = factor[0] if isinstance(factor, tuple) else factor
        by_factor.append(
            {
                "factor": factor_name,
                "horizons": frame.height,
                "first_abs_rank_ic": abs_values[0] if abs_values else None,
                "last_abs_rank_ic": abs_values[-1] if abs_values else None,
                "rank_ic_decay_ratio": decay_ratio,
                "is_monotone_decay": monotone_decay,
            }
        )

    return pl.DataFrame(by_factor)
