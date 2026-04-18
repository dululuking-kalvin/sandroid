"""Scene YAML loader.

Encoding: adjacency list. A scene file declares a root category and a flat
``categories:`` list; each entry names its ``parent_id`` (except the root).
The loader validates the V3 mode rules at load time — runtime must never see
an inconsistent tree.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from sandroid.core.domain import (
    Category,
    CategoryId,
    CategoryTree,
    Intent,
    Mode,
    OnCompleteHandoff,
    OnCompleteRules,
)


class SceneValidationError(ValueError):
    """Raised when a scene YAML fails structural or semantic validation."""


class _RawScene(BaseModel):
    """Shape of the YAML file before it becomes a CategoryTree."""

    root_id: CategoryId
    categories: list[dict[str, Any]] = Field(default_factory=list)


def load_scene(path: str | Path) -> CategoryTree:
    """Parse a scene YAML file into a validated ``CategoryTree``."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SceneValidationError(f"scene file must be a YAML mapping: {path}")
    try:
        raw = _RawScene.model_validate(data)
    except ValidationError as exc:
        raise SceneValidationError(f"invalid scene shape in {path}: {exc}") from exc
    return _build_tree(raw, source=str(path))


def load_scene_from_dict(data: dict[str, Any]) -> CategoryTree:
    """Same as :func:`load_scene` but from an in-memory dict (handy for tests)."""

    try:
        raw = _RawScene.model_validate(data)
    except ValidationError as exc:
        raise SceneValidationError(f"invalid scene shape: {exc}") from exc
    return _build_tree(raw, source="<dict>")


def _build_tree(raw: _RawScene, *, source: str) -> CategoryTree:
    by_id: dict[CategoryId, Category] = {}
    for entry in raw.categories:
        category = _parse_category(entry, source=source)
        if category.id in by_id:
            raise SceneValidationError(
                f"{source}: duplicate category id {category.id!r}",
            )
        by_id[category.id] = category

    if raw.root_id not in by_id:
        raise SceneValidationError(f"{source}: root_id {raw.root_id!r} not in categories")

    _wire_children(by_id, root_id=raw.root_id, source=source)
    _validate_structural_rules(by_id, source=source)
    _validate_unique_names_per_parent(by_id, source=source)
    _validate_extends_enum(by_id, source=source)

    return CategoryTree(root_id=raw.root_id, categories=by_id)


def _parse_category(entry: dict[str, Any], *, source: str) -> Category:
    on_complete_raw = entry.get("on_complete")
    on_complete: OnCompleteHandoff | OnCompleteRules | None = None
    if on_complete_raw is not None:
        kind = on_complete_raw.get("type")
        if kind == "handoff":
            on_complete = OnCompleteHandoff()
        elif kind == "rules":
            on_complete = OnCompleteRules.model_validate(on_complete_raw)
        else:
            raise SceneValidationError(
                f"{source}: unsupported on_complete.type {kind!r} "
                "(v0.1 ships handoff + rules only)",
            )

    intents_raw = entry.get("intents") or []
    intents = [Intent.model_validate({**item, "category_id": entry["id"]}) for item in intents_raw]

    return Category(
        id=entry["id"],
        name=entry["name"],
        parent_id=entry.get("parent_id"),
        mode=Mode(entry.get("mode", Mode.STANDARD.value)),
        extends_enum_category=entry.get("extends_enum_category"),
        intents=intents,
        on_complete=on_complete,
    )


def _wire_children(
    by_id: dict[CategoryId, Category],
    *,
    root_id: CategoryId,
    source: str,
) -> None:
    for category in by_id.values():
        if category.id == root_id:
            if category.parent_id is not None:
                raise SceneValidationError(
                    f"{source}: root category {category.id!r} must have parent_id=null",
                )
            continue
        if category.parent_id is None:
            raise SceneValidationError(
                f"{source}: non-root category {category.id!r} is missing parent_id",
            )
        parent = by_id.get(category.parent_id)
        if parent is None:
            raise SceneValidationError(
                f"{source}: category {category.id!r} points at unknown parent "
                f"{category.parent_id!r}",
            )
        parent.child_ids.append(category.id)

    _assert_acyclic(by_id, root_id=root_id, source=source)


def _assert_acyclic(
    by_id: dict[CategoryId, Category],
    *,
    root_id: CategoryId,
    source: str,
) -> None:
    seen: set[CategoryId] = set()
    stack: list[CategoryId] = [root_id]
    while stack:
        cur = stack.pop()
        if cur in seen:
            raise SceneValidationError(f"{source}: cycle detected at {cur!r}")
        seen.add(cur)
        stack.extend(by_id[cur].child_ids)
    orphans = set(by_id) - seen
    if orphans:
        raise SceneValidationError(
            f"{source}: categories unreachable from root: {sorted(orphans)}",
        )


def _validate_structural_rules(
    by_id: dict[CategoryId, Category],
    *,
    source: str,
) -> None:
    for category in by_id.values():
        props = category.mode.props
        ordinary = category.ordinary_intents
        special = category.special_intents

        if ordinary and not props.may_hold_ordinary:
            raise SceneValidationError(
                f"{source}: {category.mode.value} category {category.id!r} "
                f"may not hold ordinary intents (has {len(ordinary)})",
            )
        if special and not props.may_hold_special:
            raise SceneValidationError(
                f"{source}: {category.mode.value} category {category.id!r} "
                f"may not hold special intents (has {len(special)})",
            )
        if props.requires_special and not special:
            raise SceneValidationError(
                f"{source}: {category.mode.value} category {category.id!r} "
                "requires at least one special intent",
            )
        if props.forbids_children and category.child_ids:
            raise SceneValidationError(
                f"{source}: {category.mode.value} category {category.id!r} "
                "may not have child categories",
            )
        if props.single_child_chain and len(category.child_ids) > 1:
            raise SceneValidationError(
                f"{source}: {category.mode.value} category {category.id!r} "
                f"must be a single-child chain (has {len(category.child_ids)} children)",
            )


def _validate_unique_names_per_parent(
    by_id: dict[CategoryId, Category],
    *,
    source: str,
) -> None:
    for category in by_id.values():
        _assert_unique(
            (by_id[cid].name for cid in category.child_ids),
            context=f"children of {category.id!r}",
            source=source,
        )


def _assert_unique(names: Iterable[str], *, context: str, source: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise SceneValidationError(
                f"{source}: duplicate name {name!r} among {context} — names must be "
                "unique per parent",
            )
        seen.add(name)


def _validate_extends_enum(
    by_id: dict[CategoryId, Category],
    *,
    source: str,
) -> None:
    for category in by_id.values():
        target_id = category.extends_enum_category
        if target_id is None:
            continue
        target = by_id.get(target_id)
        if target is None:
            raise SceneValidationError(
                f"{source}: category {category.id!r} extends unknown category {target_id!r}",
            )
        if target.mode is not Mode.ENUM:
            raise SceneValidationError(
                f"{source}: extends_enum_category target {target_id!r} must be ENUM "
                f"(got {target.mode.value})",
            )
        if category.mode is Mode.ENUM:
            raise SceneValidationError(
                f"{source}: ENUM category {category.id!r} may not itself "
                "declare extends_enum_category",
            )
