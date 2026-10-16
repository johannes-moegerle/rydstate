"""Templates for the short labels of angular kets.

A template describes a short label and is used both to create the label from the quantum numbers
(:func:`format_short_label`) and to parse a label back into quantum numbers (:func:`parse_short_label`).

The template syntax is:

- ``{name:type}`` is a field, which shows the quantum number ``name`` formatted according to ``type``
  (see :data:`FIELD_TYPES`), e.g. ``{j_c:qn}`` -> ``1/2``.
  If the value of a field is a string, it is inserted as is.
  If the value of a required field is None, the placeholder of its type is shown (e.g. "n" for the principal quantum
  number), which is parsed back as None.
- ``[...]`` is an optional part, it is omitted if any of its fields is omitted (or None).
  Optional parts cannot be nested.
- All other characters are literal text.

For example, the template ``"{n:n}{l_r:l}_{j_r:qn}[ m={m:qn}]"`` creates and parses labels like ``"np_3/2"``,
``"60p_3/2"`` or ``"np_3/2 m=-1/2"``.
"""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING, NamedTuple

from rydstate.angular.utils import (
    ORBITAL_TO_STRING,
    format_quantum_number,
    get_spectroscopic_letter,
    parse_quantum_number,
    parse_spectroscopic_letter,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping


class FieldType(NamedTuple):
    regex: str
    format: Callable[[float], str]
    parse: Callable[[str], float | str]
    placeholder: str | None = None
    """Shown for a required field without value (None) and parsed back as None."""


_LETTERS = "".join(ORBITAL_TO_STRING.values())

FIELD_TYPES: dict[str, FieldType] = {
    # integer, e.g. the principal quantum number n_c
    "int": FieldType(r"\d+", lambda value: str(int(value)), int),
    # principal quantum number of the rydberg electron, with the placeholder "n" if it is not known
    "n": FieldType(r"\d+", lambda value: str(int(value)), int, placeholder="n"),
    # (lowercase) spectroscopic letter of an orbital quantum number, e.g. l_r=1 -> "p"
    "l": FieldType(
        rf"[{_LETTERS}]|\(l=\d+\)", lambda value: get_spectroscopic_letter(int(value)), parse_spectroscopic_letter
    ),
    # uppercase spectroscopic letter of an orbital quantum number, e.g. l_tot=1 -> "P"
    "L": FieldType(
        rf"[{_LETTERS.upper()}]|\(L=\d+\)",
        lambda value: get_spectroscopic_letter(int(value)).upper(),
        parse_spectroscopic_letter,
    ),
    # spin multiplicity 2s+1 of a spin quantum number, e.g. s_tot=1 -> "3"
    "mult": FieldType(r"\d+", lambda value: str(int(2 * value + 1)), lambda string: (int(string) - 1) / 2),
    # integer or half-integer quantum number, e.g. j_r=1.5 -> "3/2"
    "qn": FieldType(r"-?\d+(?:/2|\.5)?", format_quantum_number, parse_quantum_number),
    # parity, i.e. -1 or +1
    "parity": FieldType(r"[+-]1", lambda value: "+1" if value > 0 else "-1", int),
    # free text label (e.g. of a ket with unknown quantum numbers)
    "label": FieldType(r".+?", str, str),
}


class Field(NamedTuple):
    name: str
    type: FieldType


@cache
def tokenize(template: str) -> tuple[tuple[tuple[Field | str, ...], bool], ...]:
    """Split a template into a tuple of (parts, is_optional) groups.

    The result is cached, since the same few templates are used for all kets.
    """
    groups: list[tuple[list[Field | str], bool]] = [([], False)]
    pos = 0
    token_pattern = re.compile(r"\{(?P<name>\w+):(?P<type>\w+)\}|(?P<open>\[)|(?P<close>\])|(?P<text>[^{}\[\]]+)")
    for match in token_pattern.finditer(template):
        if match.start() != pos:
            break
        pos = match.end()
        is_optional = groups[-1][1]
        if match["open"] is not None:
            if is_optional:
                raise ValueError(f"Nested optional parts are not supported in short label template {template!r}.")
            groups.append(([], True))
        elif match["close"] is not None:
            if not is_optional:
                raise ValueError(f"Unmatched ']' in short label template {template!r}.")
            groups.append(([], False))
        elif match["text"] is not None:
            groups[-1][0].append(match["text"])
        else:
            if match["type"] not in FIELD_TYPES:
                raise ValueError(f"Unknown field type {match['type']!r} in short label template {template!r}.")
            groups[-1][0].append(Field(match["name"], FIELD_TYPES[match["type"]]))
    if pos != len(template) or groups[-1][1]:
        raise ValueError(f"Invalid short label template {template!r}.")
    return tuple((tuple(parts), is_optional) for parts, is_optional in groups if len(parts) > 0)


def format_short_label(template: str, values: Mapping[str, float | str | None], omit: Collection[str] = ()) -> str:
    """Create a short label from a template and the values of its fields.

    Args:
        template: The short label template.
        values: The values of the fields of the template. String values are inserted as is.
        omit: Names of fields, which should be omitted, if they are part of an optional part of the template.
            Optional parts are also omitted, if the value of any of their fields is None.
            Required fields without value show the placeholder of their type.

    Returns:
        The short label.

    """
    label = ""
    for parts, is_optional in tokenize(template):
        fields = [part for part in parts if isinstance(part, Field)]
        if is_optional and any(f.name in omit or values.get(f.name) is None for f in fields):
            continue
        for part in parts:
            if isinstance(part, str):
                label += part
                continue
            value = values.get(part.name)
            if value is None:
                if part.type.placeholder is None:
                    raise ValueError(f"The value of the required field {part.name!r} of {template!r} is missing.")
                label += part.type.placeholder
            else:
                label += value if isinstance(value, str) else part.type.format(value)
    return label


def parse_short_label(template: str, label: str) -> dict[str, float | str]:
    """Parse a short label according to a template.

    Args:
        template: The short label template.
        label: The short label to parse.

    Returns:
        A dictionary with the values of all fields, that are present in the label
        (fields showing their placeholder, like "n" for the principal quantum number, are not included).

    """
    regex = ""
    field_types: dict[str, FieldType] = {}
    for parts, is_optional in tokenize(template):
        group_regex = ""
        for part in parts:
            if isinstance(part, str):
                group_regex += re.escape(part)
            else:
                field_regex = part.type.regex
                if part.type.placeholder is not None:
                    field_regex = f"{re.escape(part.type.placeholder)}|{field_regex}"
                group_regex += f"(?P<{part.name}>{field_regex})"
                field_types[part.name] = part.type
        regex += f"(?:{group_regex})?" if is_optional else group_regex

    match = re.fullmatch(regex, label)
    if match is None:
        raise ValueError(f"The short label {label!r} does not match the format {template!r}.")
    return {
        name: field_types[name].parse(string)
        for name, string in match.groupdict().items()
        if string is not None and string != field_types[name].placeholder
    }
