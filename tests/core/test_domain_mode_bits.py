"""Pin the V3 table to property bits. If a bit drifts, this test fails loudly."""

from __future__ import annotations

import pytest

from sandroid.core.domain import MODE_PROPERTIES, Mode, SearchDepth


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            Mode.STANDARD,
            {
                "may_hold_ordinary": True,
                "may_hold_special": False,
                "searches_down": True,
                "search_depth": SearchDepth.ALL,
                "expose_ordinary": True,
                "expose_special": False,
                "allow_penetration": True,
            },
        ),
        (
            Mode.ENUM,
            {
                "may_hold_ordinary": True,
                "may_hold_special": False,
                "searches_down": False,
                "search_depth": SearchDepth.NONE,
                "expose_ordinary": True,
                "expose_special": False,
                "allow_penetration": True,
            },
        ),
        (
            Mode.SEQUENTIAL,
            {
                "may_hold_ordinary": True,
                "may_hold_special": True,
                "requires_special": True,
                "searches_down": False,
                "expose_ordinary": False,
                "expose_special": True,
                "allow_penetration": False,
                "single_child_chain": True,
            },
        ),
        (
            Mode.BRANCHING,
            {
                "may_hold_ordinary": False,
                "may_hold_special": True,
                "requires_special": True,
                "searches_down": True,
                "search_depth": SearchDepth.IMMEDIATE,
                "expose_ordinary": False,
                "expose_special": True,
                "allow_penetration": False,
            },
        ),
        (
            Mode.DRILLDOWN,
            {
                "may_hold_ordinary": True,
                "may_hold_special": True,
                "requires_special": True,
                "searches_down": True,
                "search_depth": SearchDepth.ALL,
                "expose_ordinary": True,
                "expose_special": True,
                "allow_penetration": True,
            },
        ),
        (
            Mode.MEMORY,
            {
                "may_hold_ordinary": True,
                "may_hold_special": False,
                "searches_down": False,
                "expose_ordinary": False,
                "expose_special": False,
                "allow_penetration": False,
                "forbids_children": True,
            },
        ),
    ],
)
def test_mode_property_bits_match_v3_table(mode: Mode, expected: dict[str, object]) -> None:
    props = MODE_PROPERTIES[mode]
    for field, value in expected.items():
        assert getattr(props, field) == value, f"{mode}.{field}"


def test_mode_props_accessor_agrees_with_table() -> None:
    for mode in Mode:
        assert mode.props is MODE_PROPERTIES[mode]
