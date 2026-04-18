"""Candidate-pool computation for a scene Category.

Reference implementation of the recursion in CLAUDE.md::

    scope(X) = X.ordinary_intents
             + for each child C of X: collect_via_ancestor_view(C)
             + extendsEnumCategory pool (if any)
             [child recursion only if X.S1 and bounded by X.S2]

    collect_via_ancestor_view(C):
        result = []
        if C.E1: result += C.ordinary_intents
        if C.E2: result += C.special_intents
        if C.E3: result += recurse into C's children
        return result

Pure functions — no I/O, no matching. Matching/confidence scoring is the
orchestrator's job, not ours.
"""

from __future__ import annotations

from sandroid.core.domain import (
    Category,
    CategoryId,
    CategoryTree,
    Intent,
    Mode,
    SearchDepth,
)


def scope(tree: CategoryTree, scene_id: CategoryId) -> list[Intent]:
    """Return the full candidate Intent pool for ``scene_id``.

    Raises ``ValueError`` if the requested scene is an ENUM node (ENUM
    categories only contribute via ``extends_enum_category`` references).
    """

    scene = tree.get(scene_id)
    if scene.mode is Mode.ENUM:
        raise ValueError(
            f"category {scene_id!r} is ENUM and cannot be used as a scene directly",
        )

    props = scene.mode.props
    candidates: list[Intent] = list(scene.ordinary_intents)
    # A scene's own special intents are not part of the candidate pool — they
    # exist to be matched by an ancestor, not by the node itself.

    if props.searches_down:
        for child_id in scene.child_ids:
            child = tree.get(child_id)
            candidates.extend(
                _collect_via_ancestor_view(
                    tree,
                    child,
                    remaining_depth=_initial_depth(props.search_depth),
                ),
            )

    if scene.extends_enum_category is not None:
        enum_root = tree.get(scene.extends_enum_category)
        candidates.extend(_collect_enum_subtree(tree, enum_root))

    return candidates


def _initial_depth(depth: SearchDepth) -> int | None:
    """Translate a SearchDepth into a numeric budget.

    - ``IMMEDIATE`` → 1 (only the direct child contributes)
    - ``ALL`` → ``None`` (no bound)
    - ``NONE`` → never reached; caller gates on ``searches_down`` first
    """

    if depth is SearchDepth.IMMEDIATE:
        return 1
    if depth is SearchDepth.ALL:
        return None
    return 0


def _collect_via_ancestor_view(
    tree: CategoryTree,
    category: Category,
    *,
    remaining_depth: int | None,
) -> list[Intent]:
    """Pull what an ancestor's search should see at ``category`` and below."""

    if remaining_depth is not None and remaining_depth <= 0:
        return []

    props = category.mode.props
    out: list[Intent] = []
    if props.expose_ordinary:
        out.extend(category.ordinary_intents)
    if props.expose_special:
        out.extend(category.special_intents)

    if props.allow_penetration:
        next_budget = None if remaining_depth is None else remaining_depth - 1
        if next_budget is None or next_budget > 0:
            for child_id in category.child_ids:
                child = tree.get(child_id)
                out.extend(
                    _collect_via_ancestor_view(
                        tree,
                        child,
                        remaining_depth=next_budget,
                    ),
                )

    # An ENUM reached via extendsEnumCategory joins through _collect_enum_subtree.
    # A non-ENUM category that carries its own extends pointer also drags that
    # subtree in when it is exposed to an ancestor — matches intuition that the
    # enum is "part of" the category's answerable surface.
    if category.extends_enum_category is not None and (
        props.expose_ordinary or props.expose_special
    ):
        enum_root = tree.get(category.extends_enum_category)
        out.extend(_collect_enum_subtree(tree, enum_root))

    return out


def _collect_enum_subtree(tree: CategoryTree, enum_root: Category) -> list[Intent]:
    """ENUM subtree contribution.

    ENUM nodes expose their ordinary intents and (per the V3 table)
    ``allow_penetration=True``, so the recursion walks all descendants.
    """

    if enum_root.mode is not Mode.ENUM:
        raise ValueError(
            f"extends_enum_category must point at an ENUM node; "
            f"got {enum_root.id!r} with mode={enum_root.mode.value}",
        )
    return _collect_via_ancestor_view(tree, enum_root, remaining_depth=None)
