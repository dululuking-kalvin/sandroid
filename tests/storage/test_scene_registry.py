from __future__ import annotations

from pathlib import Path

import pytest

from sandroid.storage.scene_registry import SceneRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENES_DIR = PROJECT_ROOT / "configs" / "scenes"


def test_registry_loads_example_scene() -> None:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    assert "example_bank" in registry.list_scenes()


def test_registry_resolve_by_scene_id() -> None:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    scene_id, category_id = registry.resolve(scene_id="example_bank")
    assert scene_id == "example_bank"
    assert category_id == "root"


def test_registry_resolve_by_category_path() -> None:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    scene_id, category_id = registry.resolve(category_path="ROOT->信用卡业务->挂失流程")
    assert scene_id == "example_bank"
    assert category_id == "card_lost_flow"


def test_registry_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SceneRegistry.from_directory(tmp_path / "nope")


def test_registry_rejects_unknown_scene_id() -> None:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    with pytest.raises(KeyError):
        registry.resolve(scene_id="ghost")


def test_registry_requires_scene_or_path() -> None:
    registry = SceneRegistry.from_directory(SCENES_DIR)
    with pytest.raises(ValueError, match="scene_id or category_path"):
        registry.resolve()
