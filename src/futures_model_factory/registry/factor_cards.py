from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

FactorStatus = Literal["candidate", "observation", "approved", "retired"]
FACTOR_CARD_STATUSES = {"candidate", "observation", "approved", "retired"}


@dataclass(frozen=True)
class FactorCard:
    name: str
    hypothesis: str
    mechanism_type: str
    expected_sign: str
    inputs: list[str]
    failure_conditions: list[dict[str, Any]]
    kill_conditions: list[str]
    search_context: dict[str, Any]
    rent_class: str | None = None
    payer: str | None = None
    payer_constraint: str | None = None
    transfer_mechanism: str | None = None
    our_action: str | None = None
    risk_we_take: str | None = None
    capacity_boundary: str | None = None
    observable_proxies: list[str] | None = None
    deep_attribution: dict[str, Any] | None = None
    log_contrast_proxy: dict[str, Any] | None = None
    implementation: dict[str, str] | None = None
    data_granularity: str | None = None
    mechanism_family: str | None = None
    signal_timestamp: str | None = None
    timing_contract: dict[str, Any] | None = None
    calculation_contract: dict[str, Any] | None = None
    status: FactorStatus = "candidate"


def load_factor_card(path: str | Path) -> FactorCard:
    """Load and validate a single factor card YAML."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("factor card must be a mapping")
    required = {
        "name",
        "hypothesis",
        "mechanism_type",
        "expected_sign",
        "inputs",
        "failure_conditions",
        "kill_conditions",
        "search_context",
        "deep_attribution",
        "log_contrast_proxy",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"factor card missing required fields: {sorted(missing)}")
    if len(raw["failure_conditions"]) < 3:
        raise ValueError("factor card must declare at least 3 failure conditions")
    if not isinstance(raw["deep_attribution"], dict):
        raise ValueError("factor card deep_attribution must be a mapping")
    if "market_structure_parent" not in raw["deep_attribution"]:
        raise ValueError(
            "factor card deep_attribution must declare market_structure_parent"
        )
    if not isinstance(raw["log_contrast_proxy"], dict):
        raise ValueError("factor card log_contrast_proxy must be a mapping")
    status = str(raw.get("status", "candidate"))
    if status not in FACTOR_CARD_STATUSES:
        raise ValueError(
            f"factor card status must be one of {sorted(FACTOR_CARD_STATUSES)}"
        )
    return FactorCard(
        name=raw["name"],
        hypothesis=raw["hypothesis"],
        mechanism_type=raw["mechanism_type"],
        expected_sign=str(raw["expected_sign"]),
        inputs=list(raw["inputs"]),
        failure_conditions=list(raw["failure_conditions"]),
        kill_conditions=list(raw["kill_conditions"]),
        search_context=dict(raw["search_context"]),
        rent_class=raw.get("rent_class"),
        payer=raw.get("payer"),
        payer_constraint=raw.get("payer_constraint"),
        transfer_mechanism=raw.get("transfer_mechanism"),
        our_action=raw.get("our_action"),
        risk_we_take=raw.get("risk_we_take"),
        capacity_boundary=raw.get("capacity_boundary"),
        observable_proxies=list(raw["observable_proxies"])
        if "observable_proxies" in raw
        else None,
        deep_attribution=dict(raw["deep_attribution"]),
        log_contrast_proxy=dict(raw["log_contrast_proxy"]),
        implementation=dict(raw["implementation"]) if "implementation" in raw else None,
        data_granularity=raw.get("data_granularity"),
        mechanism_family=raw.get("mechanism_family"),
        signal_timestamp=raw.get("signal_timestamp"),
        timing_contract=dict(raw["timing_contract"])
        if "timing_contract" in raw
        else None,
        calculation_contract=dict(raw["calculation_contract"])
        if "calculation_contract" in raw
        else None,
        status=status,
    )


def load_factor_cards(root: str | Path) -> list[FactorCard]:
    """Load all YAML factor cards from a registry directory."""
    base = Path(root)
    return [load_factor_card(path) for path in sorted(base.glob("*.yaml"))]
