"""Scene loader: happy path + each validation rule gets an explicit test."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sandroid.core.scene_loader import (
    SceneValidationError,
    load_scene,
    load_scene_from_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCENE = PROJECT_ROOT / "configs" / "scenes" / "example_bank.yaml"


def _minimal_tree(**overrides: Any) -> dict[str, Any]:
    """Two-node STANDARD tree; override/mutate for negative tests."""

    base: dict[str, Any] = {
        "root_id": "root",
        "categories": [
            {
                "id": "root",
                "name": "ROOT",
                "mode": "STANDARD",
                "intents": [{"id": "hi", "question": ["hi"], "answer": "hello"}],
            },
            {
                "id": "child",
                "name": "Child",
                "parent_id": "root",
                "mode": "STANDARD",
                "intents": [{"id": "c1", "question": ["c"], "answer": "c!"}],
            },
        ],
    }
    base.update(overrides)
    return base


def test_example_scene_loads() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    assert tree.root_id == "root"
    assert {"root", "card", "card_lost_flow", "card_lost_confirm", "transfer"}.issubset(
        tree.categories,
    )
    assert tree.path_of("card_lost_confirm") == "ROOT->信用卡业务->挂失流程->挂失确认"


def test_resolve_path_roundtrip() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    assert tree.resolve_path("ROOT->转账->境内转账").id == "transfer_domestic"


def test_resolve_path_rejects_unknown_name() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    with pytest.raises(KeyError):
        tree.resolve_path("ROOT->不存在的分类")


def test_branching_with_ordinary_intent_is_rejected() -> None:
    bad = _minimal_tree()
    bad["categories"][0]["mode"] = "BRANCHING"
    bad["categories"][0]["intents"] = [
        {"id": "ord", "question": ["o"], "answer": "o"},
        {"id": "spec", "question": ["s"], "is_special": True},
    ]
    with pytest.raises(SceneValidationError, match="may not hold ordinary intents"):
        load_scene_from_dict(bad)


def test_sequential_without_special_intent_is_rejected() -> None:
    bad = _minimal_tree()
    bad["categories"][1]["mode"] = "SEQUENTIAL"
    bad["categories"][1]["intents"] = [{"id": "o", "question": ["o"], "answer": "o"}]
    with pytest.raises(SceneValidationError, match="requires at least one special intent"):
        load_scene_from_dict(bad)


def test_sequential_with_two_children_is_rejected() -> None:
    bad: dict[str, Any] = {
        "root_id": "root",
        "categories": [
            {"id": "root", "name": "ROOT", "mode": "STANDARD"},
            {
                "id": "flow",
                "name": "Flow",
                "parent_id": "root",
                "mode": "SEQUENTIAL",
                "intents": [{"id": "hook", "question": ["h"], "is_special": True}],
            },
            {
                "id": "a",
                "name": "A",
                "parent_id": "flow",
                "mode": "STANDARD",
                "intents": [{"id": "ao", "question": ["a"], "answer": "a"}],
            },
            {
                "id": "b",
                "name": "B",
                "parent_id": "flow",
                "mode": "STANDARD",
                "intents": [{"id": "bo", "question": ["b"], "answer": "b"}],
            },
        ],
    }
    with pytest.raises(SceneValidationError, match="single-child chain"):
        load_scene_from_dict(bad)


def test_memory_with_children_is_rejected() -> None:
    bad = _minimal_tree()
    bad["categories"][0]["mode"] = "MEMORY"
    bad["categories"][0]["intents"] = [
        {"id": "ord", "question": ["o"], "answer": "o"},
    ]
    with pytest.raises(SceneValidationError, match="may not have child categories"):
        load_scene_from_dict(bad)


def test_duplicate_name_under_same_parent_is_rejected() -> None:
    bad = _minimal_tree()
    bad["categories"].append(
        {
            "id": "child2",
            "name": "Child",  # same name as existing child — collision
            "parent_id": "root",
            "mode": "STANDARD",
            "intents": [{"id": "x", "question": ["x"], "answer": "x"}],
        },
    )
    with pytest.raises(SceneValidationError, match="duplicate name"):
        load_scene_from_dict(bad)


def test_duplicate_category_id_is_rejected() -> None:
    bad = _minimal_tree()
    bad["categories"].append(
        {
            "id": "child",  # dup
            "name": "Child2",
            "parent_id": "root",
            "mode": "STANDARD",
            "intents": [{"id": "x", "question": ["x"], "answer": "x"}],
        },
    )
    with pytest.raises(SceneValidationError, match="duplicate category id"):
        load_scene_from_dict(bad)


def test_extends_enum_target_must_be_enum() -> None:
    bad = _minimal_tree()
    bad["categories"][1]["extends_enum_category"] = "root"  # STANDARD, not ENUM
    with pytest.raises(SceneValidationError, match="must be ENUM"):
        load_scene_from_dict(bad)


def test_extends_enum_unknown_target() -> None:
    bad = _minimal_tree()
    bad["categories"][1]["extends_enum_category"] = "ghost"
    with pytest.raises(SceneValidationError, match="unknown category"):
        load_scene_from_dict(bad)


def test_missing_parent_rejected() -> None:
    bad: dict[str, Any] = {
        "root_id": "root",
        "categories": [
            {"id": "root", "name": "ROOT", "mode": "STANDARD"},
            {
                "id": "orphan",
                "name": "Orphan",
                "parent_id": "nobody",
                "mode": "STANDARD",
                "intents": [{"id": "x", "question": ["x"], "answer": "x"}],
            },
        ],
    }
    with pytest.raises(SceneValidationError, match="unknown parent"):
        load_scene_from_dict(bad)


def test_unknown_on_complete_type_rejected() -> None:
    bad = _minimal_tree()
    bad["categories"][1]["on_complete"] = {"type": "intent_classifier"}
    with pytest.raises(SceneValidationError, match="handoff"):
        load_scene_from_dict(bad)
