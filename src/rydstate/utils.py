from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    T = TypeVar("T", bound=type)


def get_all_subclasses(cls: T, species: str | None = None, tag: str | None = None) -> list[T]:
    """Get all subclasses of cls for the given species (and tag)."""
    subclasses: list[T] = []
    direct_subclasses: list[Any] = cls.__subclasses__()
    for subclass in direct_subclasses:
        subclasses.extend(get_all_subclasses(subclass, species, tag))
        if inspect.isabstract(subclass) or getattr(subclass, "species", None) is None:
            continue
        if species is not None and getattr(subclass, "species", None) != species:
            continue
        if tag is not None and getattr(subclass, "tag", None) != tag:
            continue
        subclasses.append(subclass)

    return subclasses
