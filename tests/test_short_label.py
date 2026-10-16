from __future__ import annotations

import pytest
from rydstate.angular.short_label import format_short_label, parse_short_label


@pytest.mark.parametrize(
    ("template", "values", "omit", "label"),
    [
        ("{a:qn} [b={b:qn}]", {"a": 1.5, "b": 2}, (), "3/2 b=2"),
        ("{a:qn} [b={b:qn}]", {"a": 1.5, "b": None}, (), "3/2 "),
        ("{a:qn} [b={b:qn}]", {"a": 1.5, "b": 2}, ("b",), "3/2 "),
        ("{a:qn} b={b:qn}", {"a": 1.5, "b": 2}, ("b",), "3/2 b=2"),  # omit only affects optional parts
        ("{s:mult}{l:L}", {"s": 1, "l": 2}, (), "3D"),
    ],
)
def test_format_short_label(template: str, values: dict[str, float | None], omit: tuple[str], label: str) -> None:
    assert format_short_label(template, values, omit) == label


def test_format_short_label_missing_required_value() -> None:
    with pytest.raises(ValueError, match="required field"):
        format_short_label("{a:qn} {b:qn}", {"a": 1})


@pytest.mark.parametrize("template", ["{a:qn} [b={b:qn}", "{a:qn}] b", "[[{a:qn}]]", "{a:unknown_type}", "{a}"])
def test_invalid_template(template: str) -> None:
    with pytest.raises(ValueError, match="template"):
        parse_short_label(template, "1")


def test_format_short_label_placeholder() -> None:
    """A required field without value shows the placeholder of its type, string values are inserted as is."""
    template = "{n:n}{l:l}_{j:qn}"
    assert format_short_label(template, {"l": 1, "j": 1.5}) == "np_3/2"
    assert format_short_label(template, {"n": None, "l": 1, "j": 1.5}) == "np_3/2"
    assert format_short_label(template, {"n": "[57.3]", "l": 1, "j": 1.5}) == "[57.3]p_3/2"
    assert format_short_label(template, {"n": 60, "l": 1, "j": 1.5}) == "60p_3/2"


def test_parse_short_label_n() -> None:
    """The placeholder "n" is not part of the parsed values, an integer n is."""
    template = "{n:n}{l:l}_{j:qn}"
    assert parse_short_label(template, "np_3/2") == {"l": 1, "j": 1.5}
    assert parse_short_label(template, "60p_3/2") == {"n": 60, "l": 1, "j": 1.5}


def test_parse_short_label() -> None:
    template = "[{n:int}]{l:l}_{j:qn}[ m={m:qn}]"
    assert parse_short_label(template, "5d_5/2 m=-3/2") == {"n": 5, "l": 2, "j": 2.5, "m": -1.5}
    assert parse_short_label(template, "d_5/2") == {"l": 2, "j": 2.5}
