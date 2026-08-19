from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from rydstate.angular.utils import is_unknown, try_trivial_spin_addition

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rydstate.angular.utils import Unknown


class CoreKet:
    __slots__ = ("i_c", "n_c", "s_c", "l_c", "j_c", "f_c", "label", "quantum_numbers")

    quantum_number_names: ClassVar = ("i_c", "n_c", "s_c", "l_c", "j_c", "f_c")

    i_c: float
    """Nuclear spin quantum number."""
    n_c: int | Unknown | None
    """Principal quantum number of the core electron, None if not applicable."""
    s_c: float
    """Core electron spin quantum number (0 for alkali atoms, 0.5 for alkaline earth atoms)."""
    l_c: int | Unknown
    """Core electron orbital quantum number (usually 0)."""
    j_c: float | Unknown
    """Total core electron angular quantum number (s_c + l_c)."""
    f_c: float | Unknown
    """Total core angular quantum number (j_c + i_c)."""

    def __init__(
        self,
        i_c: float | None = None,
        n_c: Unknown | int | None = None,
        s_c: float = 0.5,
        l_c: Unknown | int | None = None,
        j_c: Unknown | float | None = None,
        f_c: Unknown | float | None = None,
        label: Unknown | str | None = None,
    ) -> None:
        """Initialize the core angular ket."""
        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set.")
        self.i_c = float(i_c)

        self.s_c = float(s_c)
        self.n_c = n_c
        if self.n_c is None and self.s_c != 0:
            raise ValueError("Principal quantum number n_c must be set if a core electron is present.")

        if l_c is None:
            raise ValueError("Core orbital angular momentum l_c must be set.")
        self.l_c = int(l_c) if not is_unknown(l_c) else l_c

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c)
        self.f_c = try_trivial_spin_addition(self.j_c, self.i_c, f_c)
        self.quantum_numbers = tuple(getattr(self, qn) for qn in self.quantum_number_names)

        self.label = label

        if any(is_unknown(x) for x in (self.l_c, self.j_c, self.f_c)) and label is None:
            raise ValueError("Label must be set if any of l_c, j_c, or f_c is unknown.")

    def __repr__(self) -> str:
        args = f"i_c={self.i_c}, s_c={self.s_c}, l_c={self.l_c}, j_c={self.j_c}, f_c={self.f_c}"
        if self.n_c is not None:
            args += f", n_c={self.n_c}"
        if self.label is not None:
            args += f", label={self.label}"
        return f"{self.__class__.__name__}({args})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoreKet):
            return NotImplemented
        if self.label != other.label:
            return False
        return self.quantum_numbers == other.quantum_numbers

    def __hash__(self) -> int:
        return hash((str(type(self)), self.quantum_numbers, self.label))

    @property
    def contains_unknown(self) -> bool:
        """Return True if any of the quantum numbers is Unknown."""
        return any(is_unknown(qn) for qn in self.quantum_numbers)

    def find_matching_core_ket(self, matching_core_ket_list: Iterable[CoreKet]) -> CoreKet:  # noqa: C901, PLR0912
        """Find the matching core ket in the given list of core kets.

        This means, is one core_ket in the list a broader or equal description of the same core state.
        Specifically, a core_ket from the list is a match if:

        - The quantum numbers are the same or the quantum number of the matching_core_ket (from the list) is unknown.
        - The label is the same, or the label of the matching_core_ket is unknown,
          or the label of the matching_core_ket is contained in the label of self.

        Args:
            matching_core_ket_list: List of core kets to search for a match.

        Returns:
            The matching core ket from the list.

        """
        matching_kets = []
        for match in matching_core_ket_list:
            if not all(
                getattr(self, qn) == getattr(match, qn) or is_unknown(getattr(match, qn))
                for qn in self.quantum_number_names
            ):
                continue
            if is_unknown(self.label) or is_unknown(match.label):
                if is_unknown(match.label):
                    matching_kets.append(match)
                continue
            if self.label is None or match.label is None:
                if self.label == match.label:
                    matching_kets.append(match)
                continue
            if match.label in self.label:
                matching_kets.append(match)
        if len(matching_kets) == 0:
            raise ValueError(f"No matching core ket found for {self} in the provided list.")
        if len(matching_kets) == 1:
            return matching_kets[0]

        # else len(matching_kets) > 1:
        # if multiple matches are found, return the most specific one
        # i.e. the one which should have all the others as matches as well
        for match in matching_kets:
            try:
                for other_core_ket in matching_kets:
                    match.find_matching_core_ket([other_core_ket])
            except ValueError:  # noqa: PERF203
                continue
            else:
                return match
        raise ValueError(
            f"Multiple matching core kets found for {self} in the provided list, "
            "but no most specific match could be identified."
        )
