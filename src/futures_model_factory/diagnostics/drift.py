from __future__ import annotations

from typing import Any, cast

import numpy as np
import polars as pl
from scipy import stats


def distribution_drift_summary(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    feature_cols: list[str],
    n_bins: int = 10,
) -> pl.DataFrame:
    """Compute PSI and KS drift statistics for named subsystem inputs/signals."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    rows: list[dict[str, float | str | int]] = []
    for feature in feature_cols:
        if feature not in reference.columns or feature not in current.columns:
            raise ValueError(f"missing drift feature: {feature}")
        ref = _finite_array(reference.get_column(feature))
        cur = _finite_array(current.get_column(feature))
        if ref.size == 0 or cur.size == 0:
            rows.append(
                {
                    "feature": feature,
                    "psi": float("nan"),
                    "wasserstein": float("nan"),
                    "ks_stat": float("nan"),
                    "ks_pvalue": float("nan"),
                    "n_ref": int(ref.size),
                    "n_cur": int(cur.size),
                }
            )
            continue
        edges = _quantile_edges(ref, n_bins)
        ref_share = _histogram_share(ref, edges)
        cur_share = _histogram_share(cur, edges)
        psi = float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))
        ks = cast(Any, stats.ks_2samp(ref, cur))
        wasserstein = stats.wasserstein_distance(ref, cur)
        rows.append(
            {
                "feature": feature,
                "psi": psi,
                "wasserstein": float(wasserstein),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "n_ref": int(ref.size),
                "n_cur": int(cur.size),
            }
        )
    return pl.DataFrame(rows)


def drift_gate_summary(
    drift: pl.DataFrame,
    *,
    psi_threshold: float | None = None,
    ks_threshold: float | None = None,
) -> pl.DataFrame:
    """Emit table-form drift gate decisions."""
    if psi_threshold is None and ks_threshold is None:
        raise ValueError("at least one threshold is required")
    exprs = []
    if psi_threshold is not None:
        exprs.append(
            (pl.col("psi") > psi_threshold).fill_null(True).alias("psi_breach")
        )
    if ks_threshold is not None:
        exprs.append(
            (pl.col("ks_stat") > ks_threshold).fill_null(True).alias("ks_breach")
        )
    gated = drift.with_columns(exprs)
    breach_cols = [
        column for column in ("psi_breach", "ks_breach") if column in gated.columns
    ]
    return gated.with_columns(
        pl.any_horizontal([pl.col(column) for column in breach_cols]).alias("breach")
    )


def _finite_array(series: pl.Series) -> np.ndarray:
    values = series.cast(pl.Float64).drop_nulls().to_numpy()
    return values[np.isfinite(values)]


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        center = float(edges[0]) if edges.size else 0.0
        edges = np.array([center - 0.5, center + 0.5])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _histogram_share(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    shares = counts.astype(float) / max(float(values.size), 1.0)
    return np.clip(shares, 1e-6, 1.0)
