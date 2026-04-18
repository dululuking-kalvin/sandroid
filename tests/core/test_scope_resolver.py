"""Scope resolver: verify each mode-interaction row from CLAUDE.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sandroid.core.scene_loader import load_scene, load_scene_from_dict
from sandroid.core.scope import scope

EXAMPLE_SCENE = Path(__file__).resolve().parents[2] / "configs" / "scenes" / "example_bank.yaml"


def _ids(intents: list[Any]) -> set[str]:
    return {i.id for i in intents}


def test_root_sees_drilldown_ordinary_and_special_through_penetration() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "root"))
    # Root's own ordinary
    assert "greeting" in ids
    # DRILLDOWN child exposes ordinary + special + penetrates further
    assert "card_lost_report" in ids
    assert "card_hook" in ids
    # SEQUENTIAL below DRILLDOWN exposes special only (its ordinary intents
    # stay internal to the flow), so the flow's own hook is visible
    assert "card_lost_flow_hook" in ids
    # MEMORY terminal node is not exposed upward (E1=E2=False)
    assert "card_lost_standard" not in ids
    assert "card_lost_expedite" not in ids


def test_branching_only_exposes_hooks_to_ancestor() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "root"))
    # BRANCHING hooks visible
    assert "transfer_domestic_hook" in ids
    assert "transfer_cross_border_hook" in ids
    # But BRANCHING blocks penetration, so the STANDARD children's own
    # ordinary intents are NOT in the root's pool
    assert "transfer_domestic_info" not in ids
    assert "transfer_cross_border_info" not in ids


def test_transfer_as_scene_searches_immediate_children_only() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "transfer"))
    # BRANCHING as a scene searches immediate children; STANDARD child exposes
    # its ordinary intents.
    assert "transfer_domestic_info" in ids
    assert "transfer_cross_border_info" in ids


def test_sequential_as_scene_does_not_search_down() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "card_lost_flow"))
    # SEQUENTIAL's own ordinary is empty here; own special is NOT in pool
    # (specials exist to be matched from above)
    assert "card_lost_flow_hook" not in ids
    # MEMORY child not reached
    assert "card_lost_standard" not in ids


def test_memory_as_scene_sees_only_its_own_ordinary() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "card_lost_confirm"))
    assert ids == {"card_lost_standard", "card_lost_expedite"}


def test_enum_cannot_be_used_as_scene() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    with pytest.raises(ValueError, match="ENUM"):
        scope(tree, "common_locations")


def test_extends_enum_merges_enum_subtree_into_candidate_pool() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "locations"))
    assert "location_intro" in ids  # own ordinary
    assert {"loc_beijing", "loc_shanghai"}.issubset(ids)  # pulled from ENUM


def test_drilldown_as_scene_includes_its_own_ordinary() -> None:
    tree = load_scene(EXAMPLE_SCENE)
    ids = _ids(scope(tree, "card"))
    assert "card_lost_report" in ids
    # Own special is excluded from own-pool (same rule as everyone else)
    assert "card_hook" not in ids
    # Penetrates downward: SEQUENTIAL child's special is visible
    assert "card_lost_flow_hook" in ids


def test_branching_depth_does_not_recurse_beyond_immediate_children() -> None:
    # Build an artificial BRANCHING scene whose immediate child is a
    # DRILLDOWN with a grandchild. BRANCHING depth=IMMEDIATE means the
    # grandchild's intents MUST NOT leak into the scope.
    data: dict[str, Any] = {
        "root_id": "root",
        "categories": [
            {
                "id": "root",
                "name": "ROOT",
                "mode": "BRANCHING",
                "intents": [
                    {"id": "root_hook", "question": ["h"], "is_special": True},
                ],
            },
            {
                "id": "mid",
                "name": "Mid",
                "parent_id": "root",
                "mode": "DRILLDOWN",
                "intents": [
                    {"id": "mid_ord", "question": ["m"], "answer": "m"},
                    {"id": "mid_spec", "question": ["ms"], "is_special": True},
                ],
            },
            {
                "id": "leaf",
                "name": "Leaf",
                "parent_id": "mid",
                "mode": "STANDARD",
                "intents": [{"id": "leaf_ord", "question": ["l"], "answer": "l"}],
            },
        ],
    }
    tree = load_scene_from_dict(data)
    ids = _ids(scope(tree, "root"))
    assert "mid_ord" in ids
    assert "mid_spec" in ids
    assert "leaf_ord" not in ids  # depth=IMMEDIATE stopped here


def test_standard_penetrates_through_drilldown_multiple_levels() -> None:
    data: dict[str, Any] = {
        "root_id": "root",
        "categories": [
            {
                "id": "root",
                "name": "ROOT",
                "mode": "STANDARD",
                "intents": [{"id": "r", "question": ["r"], "answer": "r"}],
            },
            {
                "id": "d1",
                "name": "D1",
                "parent_id": "root",
                "mode": "DRILLDOWN",
                "intents": [
                    {"id": "d1_ord", "question": ["d1"], "answer": "d1"},
                    {"id": "d1_spec", "question": ["d1s"], "is_special": True},
                ],
            },
            {
                "id": "d2",
                "name": "D2",
                "parent_id": "d1",
                "mode": "DRILLDOWN",
                "intents": [
                    {"id": "d2_ord", "question": ["d2"], "answer": "d2"},
                    {"id": "d2_spec", "question": ["d2s"], "is_special": True},
                ],
            },
            {
                "id": "leaf",
                "name": "Leaf",
                "parent_id": "d2",
                "mode": "STANDARD",
                "intents": [{"id": "leaf_ord", "question": ["l"], "answer": "l"}],
            },
        ],
    }
    tree = load_scene_from_dict(data)
    ids = _ids(scope(tree, "root"))
    assert {"r", "d1_ord", "d1_spec", "d2_ord", "d2_spec", "leaf_ord"} == ids
