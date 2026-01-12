from __future__ import annotations

from rydstate.angular.utils import try_trivial_spin_addition


class AngularCoreKet:
    __slots__ = ("i_c", "s_c", "l_c", "j_c", "f_c")

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | None = None,
        j_c: float | None = None,
        f_c: float | None = None,
    ) -> None:
        """Initialize the core angular ket."""
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set.")
        self.i_c = float(i_c)

        if s_c is None:
            raise ValueError("Core spin s_c must be set.")
        self.s_c = float(s_c)

        if l_c is None:
            raise ValueError("Core orbital angular momentum l_c must be set.")
        self.l_c = int(l_c)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c, "j_c")
        self.f_c = try_trivial_spin_addition(self.j_c, self.i_c, f_c, "f_c")

    def __repr__(self) -> str:
        return f"AngularCoreKet(i_c={self.i_c}, s_c={self.s_c}, l_c={self.l_c}, j_c={self.j_c}, f_c={self.f_c})"

    def __hash__(self) -> int:
        return hash((str(type(self)), self.i_c, self.s_c, self.l_c, self.j_c, self.f_c))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularCoreKet):
            return NotImplemented
        return (
            self.i_c == other.i_c
            and self.s_c == other.s_c
            and self.l_c == other.l_c
            and self.j_c == other.j_c
            and self.f_c == other.f_c
        )


class AngularCoreKetDummy(AngularCoreKet):
    """Dummy core spin ket for unknown quantum numbers."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"AngularCoreKetDummy(name={self.name})"

    def __hash__(self) -> int:
        return hash((str(type(self)), self.name))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularCoreKetDummy):
            return NotImplemented
        return self.name == other.name
