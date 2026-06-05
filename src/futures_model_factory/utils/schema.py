from __future__ import annotations

import polars as pl

DAILY_REQUIRED_COLUMNS = (
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

NUMERIC_DAILY_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


def normalize_code_expr(column: str = "code") -> pl.Expr:
    """Return a normalized futures symbol/contract code expression (uppercase and trimmed)."""
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.to_uppercase()
    )


def ensure_daily_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize and validate the minimum daily bar schema."""
    missing = [col for col in DAILY_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing required daily columns: {missing}")

    normalized = df.with_columns(
        pl.col("date").cast(pl.Date),
        normalize_code_expr("code").alias("code"),
        *[pl.col(col).cast(pl.Float64) for col in NUMERIC_DAILY_COLUMNS],
    )
    return normalized.select([*DAILY_REQUIRED_COLUMNS, *[c for c in normalized.columns if c not in DAILY_REQUIRED_COLUMNS]])


UNIVERSE_SCHEMA = {
    "date": pl.Date,
    "code": pl.String,
    "in_universe": pl.Boolean,
}

FACTOR_SCHEMA = {
    "date": pl.Date,
    "code": pl.String,
    "factor_value": pl.Float64,
}

WEIGHTS_SCHEMA = {
    "date": pl.Date,
    "code": pl.String,
    "target_weight": pl.Float64,
}

SCORE_SCHEMA = {
    "date": pl.Date,
    "code": pl.String,
    "score": pl.Float64,
    "tradable_flag": pl.Boolean,
}


def check_and_cast_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Explicitly select, cast, and validate column presence according to the schema."""
    missing = [col for col in schema if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for schema: {missing}")

    exprs = []
    for col, dtype in schema.items():
        if col == "date" and df.schema["date"] in (pl.Utf8, pl.String):
            exprs.append(pl.col("date").str.to_date().cast(dtype))
        elif col == "code" and df.schema["code"] != pl.String:
            exprs.append(normalize_code_expr(col).alias(col))
        else:
            exprs.append(pl.col(col).cast(dtype))

    return df.select(exprs)

