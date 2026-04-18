"""Domain model for sandroid — Category tree, Intents, Modes, Session.

The Mode enum is the canonical source of truth for the V3 orthogonal property
table declared in CLAUDE.md. Every mode is an assignment of A/S/E switches;
do not add behavior here that cannot be read off those bits.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator

CategoryId = str
IntentId = str
CategoryPath = str


class SearchDepth(StrEnum):
    """How far an S1-enabled scene searches into its subtree."""

    NONE = "none"
    IMMEDIATE = "immediate"
    ALL = "all"


class ModeProperties(BaseModel):
    """Orthogonal property bits for a Category mode.

    Groups:
      A — structure: what the node may hold
      S — search: how this node behaves when it IS the scene
      E — expose: how this node behaves when reached from an ancestor
    """

    model_config = {"frozen": True}

    # A group
    may_hold_ordinary: bool
    may_hold_special: bool
    requires_special: bool = False

    # S group
    searches_down: bool
    search_depth: SearchDepth

    # E group
    expose_ordinary: bool
    expose_special: bool
    allow_penetration: bool

    # Structural constraints enforced by the loader
    single_child_chain: bool = False
    forbids_children: bool = False


class Mode(StrEnum):
    """Category modes. Semantics live in ``MODE_PROPERTIES`` — read that table."""

    STANDARD = "STANDARD"
    ENUM = "ENUM"
    SEQUENTIAL = "SEQUENTIAL"
    BRANCHING = "BRANCHING"
    DRILLDOWN = "DRILLDOWN"
    MEMORY = "MEMORY"

    @property
    def props(self) -> ModeProperties:
        return MODE_PROPERTIES[self]


MODE_PROPERTIES: dict[Mode, ModeProperties] = {
    Mode.STANDARD: ModeProperties(
        may_hold_ordinary=True,
        may_hold_special=False,
        searches_down=True,
        search_depth=SearchDepth.ALL,
        expose_ordinary=True,
        expose_special=False,
        allow_penetration=True,
    ),
    Mode.ENUM: ModeProperties(
        # ENUM is never used as a scene directly. It contributes its subtree
        # when referenced via extendsEnumCategory. We flag it here for the
        # loader to reject "scene = an ENUM node" at config time.
        may_hold_ordinary=True,
        may_hold_special=False,
        searches_down=False,
        search_depth=SearchDepth.NONE,
        expose_ordinary=True,
        expose_special=False,
        allow_penetration=True,
    ),
    Mode.SEQUENTIAL: ModeProperties(
        may_hold_ordinary=True,
        may_hold_special=True,
        requires_special=True,
        searches_down=False,
        search_depth=SearchDepth.NONE,
        expose_ordinary=False,
        expose_special=True,
        allow_penetration=False,
        single_child_chain=True,
    ),
    Mode.BRANCHING: ModeProperties(
        may_hold_ordinary=False,
        may_hold_special=True,
        requires_special=True,
        searches_down=True,
        search_depth=SearchDepth.IMMEDIATE,
        expose_ordinary=False,
        expose_special=True,
        allow_penetration=False,
    ),
    Mode.DRILLDOWN: ModeProperties(
        may_hold_ordinary=True,
        may_hold_special=True,
        requires_special=True,
        searches_down=True,
        search_depth=SearchDepth.ALL,
        expose_ordinary=True,
        expose_special=True,
        allow_penetration=True,
    ),
    Mode.MEMORY: ModeProperties(
        may_hold_ordinary=True,
        may_hold_special=False,
        searches_down=False,
        search_depth=SearchDepth.NONE,
        expose_ordinary=False,
        expose_special=False,
        allow_penetration=False,
        forbids_children=True,
    ),
}


class Slot(BaseModel):
    """Default slot schema (pragmatic minimal — extend per-scene as needed)."""

    name: str
    type: str = "string"
    value: Any | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    source: Literal["asr", "rule", "user"] = "user"


class Intent(BaseModel):
    """A knowledge point attached to a Category.

    Ordinary vs. special is a single flag; confidence decides the top-1 at
    match time — do not privilege special Intents artificially.
    """

    id: IntentId
    category_id: CategoryId
    question: list[str] = Field(default_factory=list, description="Training utterances.")
    answer: str | None = None
    slots: list[Slot] = Field(default_factory=list)
    is_special: bool = False
    follow_up: CategoryPath | None = Field(
        default=None,
        description="If matched, push the session pointer to this category on the next turn.",
    )
    switching: CategoryPath | None = Field(
        default=None,
        description="Per-intent category switch; overrides followUp when set.",
    )


class OnCompleteHandoff(BaseModel):
    type: Literal["handoff"] = "handoff"


class OnCompleteRule(BaseModel):
    when: str | None = None
    default: bool = False
    intent: IntentId


class OnCompleteRules(BaseModel):
    type: Literal["rules"] = "rules"
    rules: list[OnCompleteRule]

    @field_validator("rules")
    @classmethod
    def _at_most_one_default(cls, rules: list[OnCompleteRule]) -> list[OnCompleteRule]:
        defaults = sum(1 for r in rules if r.default)
        if defaults > 1:
            raise ValueError("on_complete.rules may declare at most one default rule")
        return rules


OnComplete = Annotated[OnCompleteHandoff | OnCompleteRules, Field(discriminator="type")]


class Category(BaseModel):
    """Node in the global Category tree."""

    id: CategoryId
    name: str
    parent_id: CategoryId | None = None
    mode: Mode = Mode.STANDARD
    extends_enum_category: CategoryId | None = None
    intents: list[Intent] = Field(default_factory=list)
    child_ids: list[CategoryId] = Field(default_factory=list)
    on_complete: OnComplete | None = None

    @property
    def ordinary_intents(self) -> list[Intent]:
        return [i for i in self.intents if not i.is_special]

    @property
    def special_intents(self) -> list[Intent]:
        return [i for i in self.intents if i.is_special]


class CategoryTree(BaseModel):
    """In-memory view of the global Category tree.

    The loader produces one of these; runtime components consume it read-only.
    """

    root_id: CategoryId
    categories: dict[CategoryId, Category]

    def get(self, category_id: CategoryId) -> Category:
        if category_id not in self.categories:
            raise KeyError(f"unknown category_id: {category_id!r}")
        return self.categories[category_id]

    def path_of(self, category_id: CategoryId) -> CategoryPath:
        parts: list[str] = []
        cur: CategoryId | None = category_id
        while cur is not None:
            node = self.get(cur)
            parts.append(node.name)
            cur = node.parent_id
        return "->".join(reversed(parts))

    def resolve_path(self, path: CategoryPath) -> Category:
        parts = [p for p in path.split("->") if p]
        if not parts:
            raise ValueError(f"empty category_path: {path!r}")
        if parts[0] != self.get(self.root_id).name:
            raise KeyError(f"category_path does not start at root: {path!r}")
        cur = self.get(self.root_id)
        for name in parts[1:]:
            matches = [self.get(cid) for cid in cur.child_ids if self.get(cid).name == name]
            if not matches:
                raise KeyError(f"no child named {name!r} under {cur.name!r}")
            if len(matches) > 1:  # pragma: no cover — loader rejects this
                raise KeyError(f"name collision under {cur.name!r}: {name!r}")
            cur = matches[0]
        return cur


class Session(BaseModel):
    """Per-call session state. Real implementation lives behind a store later."""

    session_id: str
    current_category_id: CategoryId
    slots: dict[str, Slot] = Field(default_factory=dict)
    history: list[str] = Field(default_factory=list)
    switching_history: list[CategoryId] = Field(default_factory=list)
