"""In-memory registry of loaded scenes.

Scans a directory for ``*.yaml`` scene files at startup, validates each one
via the scene loader, and exposes lookup by ``scene_id`` or root
``category_path``. Fails fast — one bad scene takes the whole boot down.
"""

from __future__ import annotations

from pathlib import Path

from sandroid.core.domain import CategoryTree
from sandroid.core.scene_loader import load_scene


class SceneRegistry:
    """Holds one ``CategoryTree`` per scene YAML under a directory."""

    def __init__(self, trees: dict[str, CategoryTree]) -> None:
        self._trees = dict(trees)

    @classmethod
    def from_directory(cls, directory: str | Path) -> SceneRegistry:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"scene directory not found: {root}")
        trees: dict[str, CategoryTree] = {}
        for path in sorted(root.glob("*.yaml")):
            scene_id = path.stem
            if scene_id in trees:
                raise ValueError(f"duplicate scene id derived from filename: {scene_id}")
            trees[scene_id] = load_scene(path)
        return cls(trees)

    def list_scenes(self) -> list[str]:
        return sorted(self._trees)

    def get(self, scene_id: str) -> CategoryTree:
        if scene_id not in self._trees:
            raise KeyError(f"unknown scene: {scene_id!r}")
        return self._trees[scene_id]

    def resolve(
        self,
        *,
        scene_id: str | None = None,
        category_path: str | None = None,
    ) -> tuple[str, str]:
        """Resolve a ``(scene_id, category_id)`` pair from caller input.

        Exactly one of ``scene_id`` + ``category_path`` forms is expected:
        either a direct ``scene_id`` (root category is the scene) plus
        optional ``category_path``, or a ``category_path`` that encodes the
        scene via its root segment.
        """

        if category_path is not None:
            # First segment of the path names the scene root (by root Category name).
            parts = [p for p in category_path.split("->") if p]
            if not parts:
                raise ValueError("category_path is empty")
            root_name = parts[0]
            matches = [
                sid for sid, tree in self._trees.items() if tree.get(tree.root_id).name == root_name
            ]
            if not matches:
                raise KeyError(f"no scene has root named {root_name!r}")
            if len(matches) > 1:
                raise KeyError(f"scene root name {root_name!r} is ambiguous")
            tree = self._trees[matches[0]]
            category = tree.resolve_path(category_path)
            return matches[0], category.id

        if scene_id is None:
            raise ValueError("either scene_id or category_path must be provided")
        tree = self.get(scene_id)
        return scene_id, tree.root_id
