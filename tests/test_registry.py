from __future__ import annotations

from pathlib import Path

import pytest

from futures_model_factory.registry.factor_cards import (
    load_factor_card,
    load_factor_cards,
)
from futures_model_factory.registry.subsystem_cards import (
    load_subsystem_card,
    load_subsystem_cards,
)


def test_load_factor_card_success(temp_registry: Path) -> None:
    """验证合规的因子卡片能被顺利加载，并且各个字段解析正确。"""
    card_path = temp_registry / "factors" / "factor_volume_momentum.yaml"
    card = load_factor_card(card_path)

    assert card.name == "factor_volume_momentum"
    assert card.status == "candidate"
    assert card.expected_sign == "1"
    assert len(card.inputs) == 2
    assert "close_price" in card.inputs
    assert len(card.failure_conditions) == 3
    assert card.deep_attribution is not None
    assert card.deep_attribution["market_structure_parent"] == "volume_climax"


def test_load_factor_card_missing_fields(temp_registry: Path) -> None:
    """验证当因子卡片缺失必填字段时，会抛出 ValueError 错误。"""
    card_path = temp_registry / "factors" / "factor_bad.yaml"
    with pytest.raises(ValueError, match="factor card missing required fields"):
        load_factor_card(card_path)


def test_load_factor_cards_batch(temp_registry: Path) -> None:
    """验证批量加载函数 load_factor_cards。"""
    factors_dir = temp_registry / "factors"
    # 其中一个好，另一个坏（坏的会抛错导致加载失败）
    with pytest.raises(ValueError):
        load_factor_cards(factors_dir)

    # 如果我们删掉坏的
    (factors_dir / "factor_bad.yaml").unlink()
    cards = load_factor_cards(factors_dir)
    assert len(cards) == 1
    assert cards[0].name == "factor_volume_momentum"


def test_load_subsystem_card_success(temp_registry: Path) -> None:
    """验证合规的子系统卡片被正确加载，并解析其主要属性。"""
    card_path = temp_registry / "subsystems" / "subsystem_ok.yaml"
    card = load_subsystem_card(card_path)

    assert card.subsystem_id == "subsystem_ok"
    assert card.input_granularity == "raw_bars"
    assert card.bottleneck_location == "input_layer"
    assert len(card.escape_dimensions) == 1
    assert card.lifecycle_stage == "draft"
    assert card.output_signal_names == ["signal_1"]


def test_load_subsystem_card_placeholder_rejection(temp_registry: Path) -> None:
    """验证子系统卡片中包含 placeholder (例如 TBD, <...>) 时会被拒签并抛错。"""
    card_path = temp_registry / "subsystems" / "subsystem_bad.yaml"
    with pytest.raises(ValueError, match="contains placeholder text"):
        load_subsystem_card(card_path)


def test_load_subsystem_cards_batch(temp_registry: Path) -> None:
    """验证批量加载子系统卡片 load_subsystem_cards，自动过滤坏模板。"""
    subsystems_dir = temp_registry / "subsystems"
    # subsystem_bad.yaml 会由于包含 TBD 抛错
    with pytest.raises(ValueError, match="contains placeholder text"):
        load_subsystem_cards(subsystems_dir)

    # 删掉坏的
    (subsystems_dir / "subsystem_bad.yaml").unlink()
    subsystems = load_subsystem_cards(subsystems_dir)
    assert len(subsystems) == 1
    assert subsystems[0].subsystem_id == "subsystem_ok"

