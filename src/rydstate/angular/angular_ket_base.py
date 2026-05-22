from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import (
    InvalidQuantumNumbersError,
    NotSet,
    Unknown,
    check_spin_addition_rule,
    get_possible_quantum_number_values,
    is_not_set,
    is_unknown,
    try_trivial_spin_addition,
)
from rydstate.angular.wigner_symbols import clebsch_gordan_6j, clebsch_gordan_9j

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from rydstate.angular.angular_state import AngularState
    from rydstate.angular.utils import AngularMomentumQuantumNumbers, CouplingScheme
    from rydstate.species import SpeciesObject

logger = logging.getLogger(__name__)


class AngularKetBase(ABC):
    """Base class for a angular ket (i.e. a simple canonical spin ketstate)."""

    # We use __slots__ to prevent dynamic attributes and make the objects immutable after initialization
    __slots__ = ("i_c", "s_c", "l_c", "s_r", "l_r", "f_tot", "m", "name", "quantum_numbers", "_initialized")

    quantum_number_names: ClassVar[tuple[AngularMomentumQuantumNumbers, ...]]
    """Names of all well defined spin quantum numbers (without the magnetic quantum number m) in this class."""

    quantum_numbers: tuple[float, ...]
    """The quantum numbers corresponding to the quantum_number_names (without the magnetic quantum number m)."""

    coupled_quantum_numbers: ClassVar[
        dict[AngularMomentumQuantumNumbers, tuple[AngularMomentumQuantumNumbers, AngularMomentumQuantumNumbers]]
    ]
    """Mapping of coupled quantum numbers to their constituent quantum numbers."""

    coupling_scheme: CouplingScheme
    """Name of the coupling scheme, e.g. 'LS', 'JJ', or 'FJ'."""

    i_c: float
    """Nuclear spin quantum number."""
    s_c: float
    """Core electron spin quantum number (0 for alkali atoms, 0.5 for alkaline earth atoms)."""
    l_c: int | Unknown
    """Core electron orbital quantum number (usually 0)."""
    s_r: float
    """Rydberg electron spin quantum number (always 0.5)."""
    l_r: int | Unknown
    """Rydberg electron orbital quantum number."""

    f_tot: float
    """Total atom angular quantum number (including nuclear, core electron and rydberg electron contributions)."""
    m: float | NotSet
    """Magnetic quantum number, which is the projection of `f_tot` onto the quantization axis.
    If NotSet, only reduced matrix elements can be calculated.
    """

    name: str | None
    """Optional name for this ket, should only be used, if the ket has Unknown quantum numbers."""

    def __init__(
        self,
        i_c: float | None,
        s_c: float | None,
        l_c: int | Unknown | None,
        s_r: float | None,
        l_r: int | Unknown | None,
        f_tot: float | None,  # noqa: ARG002
        m: float | NotSet,
        *,
        name: str | None = None,
        species: str | SpeciesObject | None,
    ) -> None:
        """Initialize the Spin ket.

        Atomic species, e.g. 'Rb87', will not be used for calculation,
        only for convenience to infer the core electron spin and nuclear spin quantum numbers.
        """
        if species is not None:
            if isinstance(species, str):
                from rydstate.species.sqdt import SpeciesObjectSQDT  # noqa: PLC0415

                species = SpeciesObjectSQDT.from_name(species.replace("_mqdt", ""))
            # use i_c = 0 for species without defined nuclear spin (-> ignore hyperfine)
            if i_c is not None and i_c != species.i_c_number:
                raise ValueError(f"Nuclear spin i_c={i_c} does not match the species {species} with i_c={species.i_c}.")
            i_c = species.i_c_number
            s_c = 0.5 * (species.number_valence_electrons - 1)

        if i_c is None:
            raise ValueError("Nuclear spin i_c must be set or a species must be given.")
        self.i_c = float(i_c)
        if s_c is None:
            raise ValueError("Core spin s_c must be set or a species must be given.")
        self.s_c = float(s_c)
        self.l_c = l_c if l_c is not None else Unknown
        if not is_unknown(self.l_c):
            self.l_c = int(self.l_c)

        self.s_r = float(s_r) if s_r is not None else Unknown
        self.l_r = l_r if l_r is not None else Unknown
        if not is_unknown(self.l_r):
            self.l_r = int(self.l_r)

        # f_tot is set in the child classes
        self.m = NotSet if is_not_set(m) else float(m)

        self.name = name

    def _post_init(self) -> None:
        self.quantum_numbers = tuple(getattr(self, qn) for qn in self.quantum_number_names)
        self._initialized = True
        self.sanity_check()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if is_unknown(self.f_tot):
            msgs.append("f_tot cannot be determined from the given quantum numbers, please specify it explicitly.")

        if self.s_c not in [0, 0.5]:
            msgs.append(f"Core spin s_c must be 0 or 1/2, but {self.s_c=}")
        if self.s_r != 0.5:
            msgs.append(f"Rydberg electron spin s_r must be 1/2, but {self.s_r=}")

        if not is_not_set(self.m) and not -self.f_tot <= self.m <= self.f_tot:
            msgs.append(f"m must be between -f_tot and f_tot, but {self.f_tot=}, {self.m=}")

        if msgs:
            msg = "\n  ".join(msgs)
            raise InvalidQuantumNumbersError(self, msg)

    def __setattr__(self, key: str, value: object) -> None:
        # We use this custom __setattr__ to make the objects immutable after initialization
        if getattr(self, "_initialized", False):
            raise AttributeError(
                f"Cannot modify attributes of immutable {self.__class__.__name__} objects after initialization."
            )
        super().__setattr__(key, value)

    def __repr__(self) -> str:
        args = ", ".join(f"{qn}={val}" for qn, val in zip(self.quantum_number_names, self.quantum_numbers, strict=True))
        if not is_not_set(self.m):
            args += f", m={self.m}"
        return f"{self.__class__.__name__}({args})"

    def __str__(self) -> str:
        return self.__repr__().replace("AngularKetBase", "").replace("AngularKet", "")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularKetBase):
            return NotImplemented
        if type(self) is not type(other):
            return False
        if self.m != other.m:
            return False
        if self.name != other.name:
            return False
        return self.quantum_numbers == other.quantum_numbers

    def __hash__(self) -> int:
        return hash(
            (
                self.quantum_number_names,
                self.quantum_numbers,
                self.m,
                self.name,
            )
        )

    def get_qn(self, qn: AngularMomentumQuantumNumbers) -> float | Unknown:
        """Get the value of a quantum number by name."""
        if qn in self.quantum_number_names:
            return getattr(self, qn)  # type: ignore [no-any-return]

        for coupled_quantum_numbers in (
            self.coupled_quantum_numbers,
            AngularKetBaseLS.coupled_quantum_numbers,
            AngularKetBaseJJ.coupled_quantum_numbers,
            AngularKetBaseFJ.coupled_quantum_numbers,
        ):
            if qn not in coupled_quantum_numbers:
                continue

            qn_1, qn_2 = coupled_quantum_numbers[qn]
            qn_1_value = self.get_qn(qn_1)
            qn_2_value = self.get_qn(qn_2)
            qn_value = try_trivial_spin_addition(qn_1_value, qn_2_value, None)
            if not is_unknown(qn_value):
                return qn_value

        return Unknown

    @overload
    def to_state(self, coupling_scheme: Literal["LS"]) -> AngularState[AngularKetBaseLS]: ...

    @overload
    def to_state(self, coupling_scheme: Literal["JJ"]) -> AngularState[AngularKetBaseJJ]: ...

    @overload
    def to_state(self, coupling_scheme: Literal["FJ"]) -> AngularState[AngularKetBaseFJ]: ...

    @overload
    def to_state(self: Self) -> AngularState[Self]: ...

    def to_state(self, coupling_scheme: CouplingScheme | None = None) -> AngularState[Any]:
        """Convert to state in the specified coupling scheme.

        Args:
            coupling_scheme: The coupling scheme to convert to (e.g. "LS", "JJ", "FJ").
                If None, the state will be a trivial state (one component) in the current coupling scheme.

        Returns:
            The angular state in the specified coupling scheme.

        """
        if coupling_scheme is None or coupling_scheme == self.coupling_scheme:
            return self._create_angular_state([1], [self])
        if coupling_scheme == "LS":
            return self._to_state_ls()
        if coupling_scheme == "JJ":
            return self._to_state_jj()
        if coupling_scheme == "FJ":
            return self._to_state_fj()
        raise ValueError(f"Unknown coupling scheme {coupling_scheme!r}.")

    def _to_state_ls(self) -> AngularState[AngularKetBaseLS]:
        """Convert a single ket to state in LS coupling."""
        kets: list[AngularKetBaseLS] = []
        coefficients: list[float] = []

        s_tot_list = get_possible_quantum_number_values(self.s_c, self.s_r, getattr(self, "s_tot", None))
        l_tot_list = get_possible_quantum_number_values(self.l_c, self.l_r, getattr(self, "l_tot", None))
        for s_tot in s_tot_list:
            for l_tot in l_tot_list:
                if not is_unknown(l_tot):
                    l_tot = int(l_tot)  # noqa: PLW2901
                j_tot_list = get_possible_quantum_number_values(s_tot, l_tot, getattr(self, "j_tot", None))
                for j_tot in j_tot_list:
                    try:
                        ls_ket = AngularKetBaseLS(
                            i_c=self.i_c,
                            s_c=self.s_c,
                            l_c=self.l_c,
                            s_r=self.s_r,
                            l_r=self.l_r,
                            s_tot=s_tot,
                            l_tot=l_tot,
                            j_tot=j_tot,
                            f_tot=self.f_tot,
                            m=self.m,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(ls_ket)
                    if coeff != 0:
                        kets.append(ls_ket)
                        coefficients.append(coeff)

        return self._create_angular_state(coefficients, kets)

    def _to_state_jj(self) -> AngularState[AngularKetBaseJJ]:
        """Convert a single ket to state in JJ coupling."""
        kets: list[AngularKetBaseJJ] = []
        coefficients: list[float] = []

        j_c_list = get_possible_quantum_number_values(self.s_c, self.l_c, getattr(self, "j_c", None))
        j_r_list = get_possible_quantum_number_values(self.s_r, self.l_r, getattr(self, "j_r", None))
        for j_c in j_c_list:
            for j_r in j_r_list:
                j_tot_list = get_possible_quantum_number_values(j_c, j_r, getattr(self, "j_tot", None))
                for j_tot in j_tot_list:
                    try:
                        jj_ket = AngularKetBaseJJ(
                            i_c=self.i_c,
                            s_c=self.s_c,
                            l_c=self.l_c,
                            s_r=self.s_r,
                            l_r=self.l_r,
                            j_c=j_c,
                            j_r=j_r,
                            j_tot=j_tot,
                            f_tot=self.f_tot,
                            m=self.m,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(jj_ket)
                    if coeff != 0:
                        kets.append(jj_ket)
                        coefficients.append(coeff)

        return self._create_angular_state(coefficients, kets)

    def _to_state_fj(self) -> AngularState[AngularKetBaseFJ]:
        """Convert a single ket to state in FJ coupling."""
        kets: list[AngularKetBaseFJ] = []
        coefficients: list[float] = []

        j_c_list = get_possible_quantum_number_values(self.s_c, self.l_c, getattr(self, "j_c", None))
        j_r_list = get_possible_quantum_number_values(self.s_r, self.l_r, getattr(self, "j_r", None))
        for j_c in j_c_list:
            f_c_list = get_possible_quantum_number_values(j_c, self.i_c, getattr(self, "f_c", None))
            for f_c in f_c_list:
                for j_r in j_r_list:
                    try:
                        fj_ket = AngularKetBaseFJ(
                            i_c=self.i_c,
                            s_c=self.s_c,
                            l_c=self.l_c,
                            s_r=self.s_r,
                            l_r=self.l_r,
                            j_c=j_c,
                            f_c=f_c,
                            j_r=j_r,
                            f_tot=self.f_tot,
                            m=self.m,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(fj_ket)
                    if coeff != 0:
                        kets.append(fj_ket)
                        coefficients.append(coeff)

        return self._create_angular_state(coefficients, kets)

    def _create_angular_state(self, coefficients: Sequence[float], kets: Sequence[AngularKetBase]) -> AngularState[Any]:
        """Create an AngularState from coefficients and kets."""
        from rydstate.angular.angular_state import AngularState  # noqa: PLC0415

        return AngularState(coefficients, kets)

    def calc_reduced_overlap(self, other: AngularKetBase) -> float:
        """Calculate the reduced overlap <self||other> (ignoring the magnetic quantum number m).

        If both kets are of the same type (=same coupling scheme), this is just a delta function
        of all spin quantum numbers.
        If the kets are of different types, the overlap is calculated using the corresponding
        Clebsch-Gordan coefficients (/ Wigner-j symbols).
        """
        if type(self) is type(other):
            return 1.0 if self == other else 0.0

        for q in set(self.quantum_number_names) & set(other.quantum_number_names):
            if self.get_qn(q) != other.get_qn(q):
                return 0.0

        kets = [self, other]

        # JJ - FJ overlaps
        if any(isinstance(s, AngularKetBaseJJ) for s in kets) and any(isinstance(s, AngularKetBaseFJ) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetBaseJJ))
            fj = next(s for s in kets if isinstance(s, AngularKetBaseFJ))
            if is_unknown(fj.j_r) or is_unknown(fj.j_c) or is_unknown(jj.j_tot) or is_unknown(fj.f_c):
                raise RuntimeError("Cannot calculate overlap between JJ and FJ ket, due to Unknown quantum numbers.")
            return clebsch_gordan_6j(fj.j_r, fj.j_c, fj.i_c, jj.j_tot, fj.f_c, fj.f_tot)

        # JJ - LS overlaps
        if any(isinstance(s, AngularKetBaseJJ) for s in kets) and any(isinstance(s, AngularKetBaseLS) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetBaseJJ))
            ls = next(s for s in kets if isinstance(s, AngularKetBaseLS))
            if (
                is_unknown(ls.l_r)
                or is_unknown(ls.l_c)
                or is_unknown(ls.l_tot)
                or is_unknown(ls.s_tot)
                or is_unknown(jj.j_r)
                or is_unknown(jj.j_c)
                or is_unknown(jj.j_tot)
            ):
                raise RuntimeError("Cannot calculate overlap between JJ and LS ket, due to Unknown quantum numbers.")
            # NOTE: it matters, whether you first put all 3 l's and then all 3 s's or the other way round
            # (see symmetry properties of 9j symbol)
            # this convention is used, such that all matrix elements work out correctly, no matter in which
            # coupling scheme they are calculated
            return clebsch_gordan_9j(ls.l_r, ls.l_c, ls.l_tot, ls.s_r, ls.s_c, ls.s_tot, jj.j_r, jj.j_c, jj.j_tot)

        # FJ - LS overlaps
        if any(isinstance(s, AngularKetBaseFJ) for s in kets) and any(isinstance(s, AngularKetBaseLS) for s in kets):
            fj = next(s for s in kets if isinstance(s, AngularKetBaseFJ))
            ls = next(s for s in kets if isinstance(s, AngularKetBaseLS))
            ov: float = 0
            for coeff, jj_ket in fj.to_state("JJ"):
                ov += coeff * ls.calc_reduced_overlap(jj_ket)
            return float(ov)

        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")


class AngularKetBaseLS(AngularKetBase):
    """Spin ket in LS coupling."""

    __slots__ = ("s_tot", "l_tot", "j_tot")
    quantum_number_names: ClassVar = ("i_c", "s_c", "l_c", "s_r", "l_r", "s_tot", "l_tot", "j_tot", "f_tot")
    coupled_quantum_numbers: ClassVar = {
        "s_tot": ("s_c", "s_r"),
        "l_tot": ("l_c", "l_r"),
        "j_tot": ("s_tot", "l_tot"),
        "f_tot": ("i_c", "j_tot"),
    }
    coupling_scheme = "LS"

    s_tot: float | Unknown
    """Total electron spin quantum number (s_c + s_r)."""
    l_tot: int | Unknown
    """Total electron orbital quantum number (l_c + l_r)."""
    j_tot: float | Unknown
    """Total electron angular momentum quantum number (s_tot + l_tot)."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = Unknown,
        s_r: float | None = 0.5,
        l_r: int | Unknown | None = Unknown,
        s_tot: float | Unknown | None = None,
        l_tot: int | Unknown | None = None,
        j_tot: float | Unknown | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        *,
        name: str | None = None,
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, name=name, species=species)

        self.s_tot = try_trivial_spin_addition(self.s_c, self.s_r, s_tot)
        self.l_tot = try_trivial_spin_addition(self.l_c, self.l_r, l_tot)  # type: ignore [assignment]
        if not is_unknown(self.l_tot):
            self.l_tot = int(self.l_tot)
        self.j_tot = try_trivial_spin_addition(self.l_tot, self.s_tot, j_tot)
        self.f_tot = try_trivial_spin_addition(self.j_tot, self.i_c, f_tot)  # type: ignore [assignment]

        self._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not check_spin_addition_rule(self.l_r, self.l_c, self.l_tot):
            msgs.append(f"{self.l_r=}, {self.l_c=}, {self.l_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.s_r, self.s_c, self.s_tot):
            msgs.append(f"{self.s_r=}, {self.s_c=}, {self.s_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.l_tot, self.s_tot, self.j_tot):
            msgs.append(f"{self.l_tot=}, {self.s_tot=}, {self.j_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_tot, self.i_c, self.f_tot):
            msgs.append(f"{self.j_tot=}, {self.i_c=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)


class AngularKetBaseJJ(AngularKetBase):
    """Spin ket in JJ coupling."""

    __slots__ = ("j_c", "j_r", "j_tot")
    quantum_number_names: ClassVar = ("i_c", "s_c", "l_c", "s_r", "l_r", "j_c", "j_r", "j_tot", "f_tot")
    coupled_quantum_numbers: ClassVar = {
        "j_c": ("s_c", "l_c"),
        "j_r": ("s_r", "l_r"),
        "j_tot": ("j_c", "j_r"),
        "f_tot": ("i_c", "j_tot"),
    }
    coupling_scheme = "JJ"

    j_c: float | Unknown
    """Total core electron angular quantum number (s_c + l_c)."""
    j_r: float | Unknown
    """Total rydberg electron angular quantum number (s_r + l_r)."""
    j_tot: float | Unknown
    """Total electron angular momentum quantum number (j_c + j_r)."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = Unknown,
        s_r: float | None = 0.5,
        l_r: int | Unknown | None = Unknown,
        j_c: float | Unknown | None = None,
        j_r: float | Unknown | None = None,
        j_tot: float | Unknown | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        *,
        name: str | None = None,
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, name=name, species=species)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c)
        self.j_r = try_trivial_spin_addition(self.l_r, self.s_r, j_r)
        self.j_tot = try_trivial_spin_addition(self.j_c, self.j_r, j_tot)
        self.f_tot = try_trivial_spin_addition(self.j_tot, self.i_c, f_tot)  # type: ignore [assignment]

        self._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not check_spin_addition_rule(self.l_c, self.s_c, self.j_c):
            msgs.append(f"{self.l_c=}, {self.s_c=}, {self.j_c=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.l_r, self.s_r, self.j_r):
            msgs.append(f"{self.l_r=}, {self.s_r=}, {self.j_r=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_c, self.j_r, self.j_tot):
            msgs.append(f"{self.j_c=}, {self.j_r=}, {self.j_tot=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_tot, self.i_c, self.f_tot):
            msgs.append(f"{self.j_tot=}, {self.i_c=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)


class AngularKetBaseFJ(AngularKetBase):
    """Spin ket in FJ coupling."""

    __slots__ = ("j_c", "f_c", "j_r")
    quantum_number_names: ClassVar = ("i_c", "s_c", "l_c", "s_r", "l_r", "j_c", "f_c", "j_r", "f_tot")
    coupled_quantum_numbers: ClassVar = {
        "j_c": ("s_c", "l_c"),
        "f_c": ("i_c", "j_c"),
        "j_r": ("s_r", "l_r"),
        "f_tot": ("f_c", "j_r"),
    }
    coupling_scheme = "FJ"

    j_c: float | Unknown
    """Total core electron angular quantum number (s_c + l_c)."""
    f_c: float | Unknown
    """Total core angular quantum number (j_c + i_c)."""
    j_r: float | Unknown
    """Total rydberg electron angular quantum number (s_r + l_r)."""

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = Unknown,
        s_r: float | None = 0.5,
        l_r: int | Unknown | None = Unknown,
        j_c: float | Unknown | None = None,
        f_c: float | Unknown | None = None,
        j_r: float | Unknown | None = None,
        f_tot: float | None = None,
        m: float | NotSet = NotSet,
        *,
        name: str | None = None,
        species: str | SpeciesObject | None = None,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, name=name, species=species)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c)
        self.j_r = try_trivial_spin_addition(self.l_r, self.s_r, j_r)
        self.f_c = try_trivial_spin_addition(self.j_c, self.i_c, f_c)
        self.f_tot = try_trivial_spin_addition(self.f_c, self.j_r, f_tot)  # type: ignore [assignment]

        self._post_init()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not check_spin_addition_rule(self.l_c, self.s_c, self.j_c):
            msgs.append(f"{self.l_c=}, {self.s_c=}, {self.j_c=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.l_r, self.s_r, self.j_r):
            msgs.append(f"{self.l_r=}, {self.s_r=}, {self.j_r=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.j_c, self.i_c, self.f_c):
            msgs.append(f"{self.j_c=}, {self.i_c=}, {self.f_c=} don't satisfy spin addition rule.")

        if not check_spin_addition_rule(self.f_c, self.j_r, self.f_tot):
            msgs.append(f"{self.f_c=}, {self.j_r=}, {self.f_tot=} don't satisfy spin addition rule.")

        super().sanity_check(msgs)

    def get_core_ket(self) -> CoreKet:
        """Get the core ket corresponding to this FJ ket."""
        return CoreKet(i_c=self.i_c, s_c=self.s_c, l_c=self.l_c, j_c=self.j_c, f_c=self.f_c)
