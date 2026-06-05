from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]

LifecycleStage = Literal["draft", "observation", "candidate", "approved", "deprecated", "rejected"]

_PLACEHOLDER_MARKERS = ("<", ">", "TBD", "TODO", "N/A?")
_INPUT_GRANULARITIES = {"raw_bars", "minute_aggregates", "factor_table", "mixed"}
_BOTTLENECK_LOCATIONS = {"input_layer", "shared_representation", "per_head_output"}
_ESCAPE_DIMENSIONS = {
    "scalar_projection",
    "per_name_independence",
    "stationary_semantics",
    "human_articulability",
    "fixed_combinator",
}
_PROFIT_SOURCE_CATEGORIES = {
    "adverse_selection",
    "liquidity_provision",
    "statistical_arbitrage",
    "behavioral_bias",
    "structural_arbitrage",
    "microstructure_mean_reversion",
    "momentum",
    "cross_asset_spillover",
    "regime_premium",
    "other",
}
_BASELINE_METRICS = {"incremental_IR", "incremental_Sharpe"}
_DRIFT_CHECKS = {"PSI", "KS", "reconstruction_error"}
_DRIFT_ACTIONS = {"downgrade_weight", "deactivate"}
_LIFECYCLE_STAGES = {"draft", "observation", "candidate", "approved", "deprecated", "rejected"}


@dataclass(frozen=True)
class OutputSignalSpec:
    name: str
    range: str
    semantics: str


@dataclass(frozen=True)
class SubsystemCard:
    subsystem_id: str
    input_granularity: str
    bottleneck_location: str
    escape_dimensions: list[str]
    interpretability_layers_kept: list[str]
    interpretability_layers_given_up: list[str]
    mechanism_class: str
    profit_source_category: str
    expected_mechanism_sign: str
    scope_conditions: dict[str, list[str]]
    baseline_to_beat: dict[str, Any]
    kill_condition: dict[str, Any]
    drift_monitoring: dict[str, Any]
    search_context: dict[str, Any]
    compensations: dict[str, str]
    output_signals: list[OutputSignalSpec]
    audit: dict[str, Any]

    @property
    def output_signal_names(self) -> list[str]:
        return [signal.name for signal in self.output_signals]

    @property
    def lifecycle_stage(self) -> LifecycleStage:
        return self.audit.get("lifecycle_stage", "draft")


def load_subsystem_card(path: str | Path) -> SubsystemCard:
    """Load and validate a Subsystem Card YAML.

    A Subsystem Card is the L1+L2 pre-registration artifact for learned or
    representation-style subsystems. This loader intentionally rejects template
    placeholders so that a copied card cannot accidentally authorize training.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("subsystem card must be a mapping")

    required = {
        "subsystem_id",
        "input_granularity",
        "bottleneck_location",
        "escape_dimensions",
        "interpretability_layers_kept",
        "interpretability_layers_given_up",
        "mechanism_class",
        "profit_source_category",
        "expected_mechanism_sign",
        "scope_conditions",
        "baseline_to_beat",
        "kill_condition",
        "drift_monitoring",
        "search_context",
        "compensations",
        "output_signals",
        "audit",
    }
    missing = required - set(raw)
    if missing:
        raise ValueError(f"subsystem card missing required fields: {sorted(missing)}")

    _reject_placeholders(raw)
    _validate_enums(raw)
    _validate_scope_conditions(raw["scope_conditions"])
    _validate_baseline(raw["baseline_to_beat"])
    _validate_kill_condition(raw["kill_condition"])
    _validate_drift_monitoring(raw["drift_monitoring"])
    _validate_search_context(raw["search_context"])
    _validate_compensations(raw["compensations"], raw["interpretability_layers_given_up"])
    output_signals = _validate_output_signals(raw["output_signals"])
    _validate_audit(raw["audit"])

    return SubsystemCard(
        subsystem_id=str(raw["subsystem_id"]),
        input_granularity=str(raw["input_granularity"]),
        bottleneck_location=str(raw["bottleneck_location"]),
        escape_dimensions=list(raw["escape_dimensions"]),
        interpretability_layers_kept=list(raw["interpretability_layers_kept"]),
        interpretability_layers_given_up=list(raw["interpretability_layers_given_up"]),
        mechanism_class=str(raw["mechanism_class"]),
        profit_source_category=str(raw["profit_source_category"]),
        expected_mechanism_sign=str(raw["expected_mechanism_sign"]),
        scope_conditions={key: list(value) for key, value in raw["scope_conditions"].items()},
        baseline_to_beat=dict(raw["baseline_to_beat"]),
        kill_condition=dict(raw["kill_condition"]),
        drift_monitoring=dict(raw["drift_monitoring"]),
        search_context=dict(raw["search_context"]),
        compensations=dict(raw["compensations"]),
        output_signals=output_signals,
        audit=dict(raw["audit"]),
    )


def load_subsystem_cards(root: str | Path) -> list[SubsystemCard]:
    """Load all non-template YAML subsystem cards from a registry directory."""
    base = Path(root)
    return [load_subsystem_card(path) for path in sorted(base.glob("*.yaml")) if not path.name.startswith("_")]


def _reject_placeholders(value: Any, *, path: str = "card") -> None:
    if value is None:
        raise ValueError(f"{path} must not be empty")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{path} must not be empty")
        if any(marker in stripped for marker in _PLACEHOLDER_MARKERS):
            raise ValueError(f"{path} contains placeholder text")
        return
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"{path} must not be empty")
        for key, item in value.items():
            _reject_placeholders(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        if not value and not path.endswith(".amendments"):
            raise ValueError(f"{path} must not be empty")
        for index, item in enumerate(value):
            _reject_placeholders(item, path=f"{path}[{index}]")


def _validate_enums(raw: dict[str, Any]) -> None:
    _require_member(raw["input_granularity"], _INPUT_GRANULARITIES, "input_granularity")
    _require_member(raw["bottleneck_location"], _BOTTLENECK_LOCATIONS, "bottleneck_location")
    for dimension in raw["escape_dimensions"]:
        _require_member(dimension, _ESCAPE_DIMENSIONS, "escape_dimensions")
    if not {"L1", "L2"}.issubset(set(raw["interpretability_layers_kept"])):
        raise ValueError("subsystem card must keep L1 and L2")
    if "L2" in set(raw["interpretability_layers_given_up"]):
        raise ValueError("subsystem card may not give up L2")
    _require_member(raw["profit_source_category"], _PROFIT_SOURCE_CATEGORIES, "profit_source_category")


def _validate_scope_conditions(scope: dict[str, Any]) -> None:
    for key in ("active_when", "inactive_when"):
        value = scope.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"scope_conditions.{key} must be a non-empty list")


def _validate_baseline(baseline: dict[str, Any]) -> None:
    for key in ("description", "metric", "threshold", "evaluation_window"):
        if key not in baseline:
            raise ValueError(f"baseline_to_beat missing {key}")
    _require_member(baseline["metric"], _BASELINE_METRICS, "baseline_to_beat.metric")
    _require_number(baseline["threshold"], "baseline_to_beat.threshold")


def _validate_kill_condition(kill_condition: dict[str, Any]) -> None:
    for key in ("metric", "threshold", "consecutive_days", "rearm_rule"):
        if key not in kill_condition:
            raise ValueError(f"kill_condition missing {key}")
    _require_number(kill_condition["threshold"], "kill_condition.threshold")
    if int(kill_condition["consecutive_days"]) <= 0:
        raise ValueError("kill_condition.consecutive_days must be positive")


def _validate_drift_monitoring(drift: dict[str, Any]) -> None:
    for key in ("input_distribution_check", "threshold", "action_on_breach"):
        if key not in drift:
            raise ValueError(f"drift_monitoring missing {key}")
    _require_member(drift["input_distribution_check"], _DRIFT_CHECKS, "drift_monitoring.input_distribution_check")
    _require_member(drift["action_on_breach"], _DRIFT_ACTIONS, "drift_monitoring.action_on_breach")
    _require_number(drift["threshold"], "drift_monitoring.threshold")


def _validate_search_context(search: dict[str, Any]) -> None:
    for key in (
        "architectures_considered",
        "encoders_considered",
        "horizons_considered",
        "selection_rule",
        "research_ledger_entries",
    ):
        if key not in search:
            raise ValueError(f"search_context missing {key}")
    if int(search["architectures_considered"]) <= 0:
        raise ValueError("search_context.architectures_considered must be positive")
    if not isinstance(search["horizons_considered"], list) or not search["horizons_considered"]:
        raise ValueError("search_context.horizons_considered must be a non-empty list")
    if not isinstance(search["research_ledger_entries"], list) or not search["research_ledger_entries"]:
        raise ValueError("search_context.research_ledger_entries must be a non-empty list")


def _validate_compensations(compensations: dict[str, Any], given_up: list[str]) -> None:
    if given_up and "for_L3_readability_surrender" not in compensations:
        raise ValueError("compensations must declare for_L3_readability_surrender")


def _validate_output_signals(signals: list[dict[str, Any]]) -> list[OutputSignalSpec]:
    if not isinstance(signals, list) or not signals:
        raise ValueError("output_signals must be a non-empty list")
    if len(signals) > 3:
        raise ValueError("subsystem card may expose at most 3 named output signals")
    names: set[str] = set()
    parsed: list[OutputSignalSpec] = []
    for signal in signals:
        for key in ("name", "range", "semantics"):
            if key not in signal:
                raise ValueError(f"output_signals entry missing {key}")
        name = str(signal["name"])
        if name in names:
            raise ValueError(f"duplicate output signal: {name}")
        if "latent" in name.lower() or "embedding" in name.lower():
            raise ValueError("output signal names must be semantic, not raw latent vectors")
        names.add(name)
        parsed.append(OutputSignalSpec(name=name, range=str(signal["range"]), semantics=str(signal["semantics"])))
    return parsed


def _validate_audit(audit: dict[str, Any]) -> None:
    stage = audit.get("lifecycle_stage", "draft")
    _require_member(stage, _LIFECYCLE_STAGES, "audit.lifecycle_stage")


def _require_member(value: object, allowed: set[str], field: str) -> None:
    if value not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}")


def _require_number(value: object, field: str) -> None:
    if not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
