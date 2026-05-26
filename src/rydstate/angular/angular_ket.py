from __future__ import annotations

import logging
from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, overload

from rydstate.angular.angular_matrix_element import (
    calc_prefactor_of_operator_in_coupled_scheme,
    calc_reduced_identity_matrix_element,
    calc_reduced_spherical_matrix_element,
    calc_reduced_spin_matrix_element,
)
from rydstate.angular.core_ket import CoreKet
from rydstate.angular.utils import (
    AllKnown,
    InvalidQuantumNumbersError,
    NotSet,
    Unknown,
    check_spin_addition_rule,
    get_possible_quantum_number_values,
    is_angular_momentum_quantum_number,
    is_angular_operator_type,
    is_not_set,
    is_unknown,
    minus_one_pow,
    try_trivial_spin_addition,
)
from rydstate.angular.wigner_symbols import calc_wigner_3j, clebsch_gordan_6j, clebsch_gordan_9j

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from rydstate.angular.angular_state import AngularState
    from rydstate.angular.utils import AngularMomentumQuantumNumbers, AngularOperatorType, CouplingScheme
    from rydstate.species import SpeciesObject

logger = logging.getLogger(__name__)

GenericT_Unknown = TypeVar("GenericT_Unknown", float, AllKnown, Unknown)
T_Unknown = TypeVar("T_Unknown", float, AllKnown, Unknown)


class AngularKetBase(ABC, Generic[GenericT_Unknown]):
    """Base class for a angular ket (i.e. a simple canonical spin ketstate)."""

    # We use __slots__ to prevent dynamic attributes and make the objects immutable after initialization
    __slots__ = (
        "i_c",
        "s_c",
        "l_c",
        "s_r",
        "l_r",
        "f_tot",
        "m",
        "name",
        "quantum_numbers",
        "_allow_unknown",
        "_initialized",
    )

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
    l_c: int | GenericT_Unknown
    """Core electron orbital quantum number (usually 0)."""
    s_r: float
    """Rydberg electron spin quantum number (always 0.5)."""
    l_r: int | GenericT_Unknown
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
        allow_unknown: bool = False,
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
        self.l_c = l_c if l_c is not None else Unknown  # type: ignore [assignment]
        if not is_unknown(self.l_c):
            self.l_c = int(self.l_c)

        self.s_r = float(s_r) if s_r is not None else Unknown
        self.l_r = l_r if l_r is not None else Unknown  # type: ignore [assignment]
        if not is_unknown(self.l_r):
            self.l_r = int(self.l_r)

        # f_tot is set in the child classes
        self.m = NotSet if is_not_set(m) else float(m)

        self.name = name

        self._allow_unknown = allow_unknown

    def _post_init(self) -> None:
        self.quantum_numbers = tuple(getattr(self, qn) for qn in self.quantum_number_names)
        self._initialized = True
        self.sanity_check()

    def sanity_check(self, msgs: list[str] | None = None) -> None:
        """Check that the quantum numbers are valid."""
        msgs = msgs if msgs is not None else []

        if not self._allow_unknown and any(is_unknown(qn) for qn in self.quantum_numbers):
            raise ValueError(f"Unknown quantum numbers detected, but allow_unknown is False for {self!r}.")

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
        return self.__repr__().replace("AngularKet", "")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AngularKetBase):
            return NotImplemented
        if not self._equal_up_to_m(other):
            return False
        return bool(self.m == other.m)

    def _equal_up_to_m(self, other: AngularKetBase[Any]) -> bool:
        if type(self) is not type(other):
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

    @property
    def contains_unknown(self) -> bool:
        """Return True if any of the quantum numbers is Unknown."""
        return any(is_unknown(qn) for qn in self.quantum_numbers)

    def get_qn(self, qn: AngularMomentumQuantumNumbers) -> float | GenericT_Unknown:
        """Get the value of a quantum number by name."""
        qn_value: float | Unknown = Unknown
        if qn in self.quantum_number_names:
            qn_value = getattr(self, qn)
        else:
            for coupled_quantum_numbers in (
                self.coupled_quantum_numbers,
                AngularKetLS.coupled_quantum_numbers,
                AngularKetJJ.coupled_quantum_numbers,
                AngularKetFJ.coupled_quantum_numbers,
            ):
                if qn not in coupled_quantum_numbers:
                    continue

                qn_1, qn_2 = coupled_quantum_numbers[qn]
                qn_1_value = self.get_qn(qn_1)
                qn_2_value = self.get_qn(qn_2)
                qn_value = try_trivial_spin_addition(qn_1_value, qn_2_value, None)
                if not is_unknown(qn_value):
                    break

        if not self._allow_unknown and is_unknown(qn_value):
            raise ValueError(f"Quantum number {qn} is unknown for {self!r}.")

        return qn_value  # type: ignore [return-value]

    def calc_exp_qn(self, qn: AngularMomentumQuantumNumbers) -> float | GenericT_Unknown:
        """Calculate the expectation value of a quantum number qn.

        If the quantum number is a good quantum number simply return it,
        otherwise calculate it, see also AngularState.calc_exp_qn for more details.

        Args:
            qn: The quantum number to calculate the expectation value for.

        """
        if qn in self.quantum_number_names:
            return self.get_qn(qn)
        return self.to_state().calc_exp_qn(qn)

    def calc_std_qn(self, qn: AngularMomentumQuantumNumbers) -> float | GenericT_Unknown:
        """Calculate the standard deviation of a quantum number qn.

        If the quantum number is a good quantum number return 0,
        otherwise calculate the std, see also AngularState.calc_std_qn for more details.

        Args:
            qn: The quantum number to calculate the standard deviation for.

        """
        if qn in self.quantum_number_names:
            if is_unknown(self.get_qn(qn)):
                return Unknown
            return 0
        return self.to_state().calc_std_qn(qn)

    @overload
    def to_state(self: Self, coupling_scheme: None = None) -> AngularState[Self]: ...

    @overload
    def to_state(
        self: AngularKetBase[T_Unknown], coupling_scheme: Literal["LS"]
    ) -> AngularState[AngularKetLS[T_Unknown]]: ...

    @overload
    def to_state(
        self: AngularKetBase[T_Unknown], coupling_scheme: Literal["JJ"]
    ) -> AngularState[AngularKetJJ[T_Unknown]]: ...

    @overload
    def to_state(
        self: AngularKetBase[T_Unknown], coupling_scheme: Literal["FJ"]
    ) -> AngularState[AngularKetFJ[T_Unknown]]: ...

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

    def _to_state_ls(self: AngularKetBase[T_Unknown]) -> AngularState[AngularKetLS[T_Unknown]]:
        """Convert a single ket to state in LS coupling."""
        if self.contains_unknown:
            s_tot = try_trivial_spin_addition(self.s_c, self.s_r, None)
            l_tot = try_trivial_spin_addition(self.l_c, self.l_r, None)
            j_tot = try_trivial_spin_addition(s_tot, l_tot, None)
            ket = AngularKetLS(  # type: ignore [call-overload,misc]
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
                name=self.name,
                allow_unknown=True,
            )
            return self._create_angular_state([1], [ket])

        kets: list[AngularKetLS[T_Unknown]] = []
        coefficients: list[float] = []

        s_tot_list = get_possible_quantum_number_values(self.s_c, self.s_r, getattr(self, "s_tot", None))
        l_tot_list = get_possible_quantum_number_values(self.l_c, self.l_r, getattr(self, "l_tot", None))
        for s_tot in s_tot_list:
            for l_tot in l_tot_list:
                j_tot_list = get_possible_quantum_number_values(s_tot, l_tot, getattr(self, "j_tot", None))
                for j_tot in j_tot_list:
                    try:
                        ls_ket = AngularKetLS(  # type: ignore [call-overload,misc]
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
                            name=self.name,
                            allow_unknown=self._allow_unknown,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(ls_ket)
                    if coeff != 0:
                        kets.append(ls_ket)
                        coefficients.append(coeff)

        return self._create_angular_state(coefficients, kets)

    def _to_state_jj(self: AngularKetBase[T_Unknown]) -> AngularState[AngularKetJJ[T_Unknown]]:
        """Convert a single ket to state in JJ coupling."""
        if self.contains_unknown:
            j_c = try_trivial_spin_addition(self.s_c, self.l_c, None)
            j_r = try_trivial_spin_addition(self.s_r, self.l_r, None)
            j_tot = try_trivial_spin_addition(j_c, j_r, None)
            ket = AngularKetJJ(  # type: ignore [call-overload,misc]
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
                name=self.name,
                allow_unknown=self._allow_unknown,
            )
            return self._create_angular_state([1], [ket])

        kets: list[AngularKetJJ[T_Unknown]] = []
        coefficients: list[float] = []

        j_c_list = get_possible_quantum_number_values(self.s_c, self.l_c, getattr(self, "j_c", None))
        j_r_list = get_possible_quantum_number_values(self.s_r, self.l_r, getattr(self, "j_r", None))
        for j_c in j_c_list:
            for j_r in j_r_list:
                j_tot_list = get_possible_quantum_number_values(j_c, j_r, getattr(self, "j_tot", None))
                for j_tot in j_tot_list:
                    try:
                        jj_ket = AngularKetJJ(  # type: ignore [call-overload,misc]
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
                            name=self.name,
                            allow_unknown=self._allow_unknown,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(jj_ket)
                    if coeff != 0:
                        kets.append(jj_ket)
                        coefficients.append(coeff)

        return self._create_angular_state(coefficients, kets)

    def _to_state_fj(self: AngularKetBase[T_Unknown]) -> AngularState[AngularKetFJ[T_Unknown]]:
        """Convert a single ket to state in FJ coupling."""
        if self.contains_unknown:
            j_c = try_trivial_spin_addition(self.s_c, self.l_c, None)
            j_r = try_trivial_spin_addition(self.s_r, self.l_r, None)
            f_c = try_trivial_spin_addition(j_c, self.i_c, None)
            ket = AngularKetFJ(  # type: ignore [call-overload,misc]
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
                name=self.name,
                allow_unknown=self._allow_unknown,
            )
            return self._create_angular_state([1], [ket])

        kets: list[AngularKetFJ[T_Unknown]] = []
        coefficients: list[float] = []

        j_c_list = get_possible_quantum_number_values(self.s_c, self.l_c, getattr(self, "j_c", None))
        j_r_list = get_possible_quantum_number_values(self.s_r, self.l_r, getattr(self, "j_r", None))
        for j_c in j_c_list:
            f_c_list = get_possible_quantum_number_values(j_c, self.i_c, getattr(self, "f_c", None))
            for f_c in f_c_list:
                for j_r in j_r_list:
                    try:
                        fj_ket = AngularKetFJ(  # type: ignore [call-overload,misc]
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
                            name=self.name,
                            allow_unknown=self._allow_unknown,
                        )
                    except InvalidQuantumNumbersError:
                        continue
                    coeff = self.calc_reduced_overlap(fj_ket)
                    if coeff != 0:
                        kets.append(fj_ket)
                        coefficients.append(coeff)

        return self._create_angular_state(coefficients, kets)

    def _create_angular_state(
        self, coefficients: Sequence[float], kets: Sequence[AngularKetBase[Any]]
    ) -> AngularState[Any]:
        """Create an AngularState from coefficients and kets."""
        from rydstate.angular.angular_state import AngularState  # noqa: PLC0415

        return AngularState(coefficients, kets)

    def calc_reduced_overlap(self, other: AngularKetBase[Any]) -> float:
        """Calculate the reduced overlap <self||other> (ignoring the magnetic quantum number m).

        If both kets are of the same type (=same coupling scheme), this is just a delta function
        of all spin quantum numbers.
        If the kets are of different types, the overlap is calculated using the corresponding
        Clebsch-Gordan coefficients (/ Wigner-j symbols).
        """
        if self.coupling_scheme == other.coupling_scheme:
            return 1.0 if self._equal_up_to_m(other) else 0.0

        for q in set(self.quantum_number_names) & set(other.quantum_number_names):
            if self.get_qn(q) != other.get_qn(q):
                return 0.0

        kets = [self, other]

        # JJ - FJ overlaps
        if any(isinstance(s, AngularKetJJ) for s in kets) and any(isinstance(s, AngularKetFJ) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetJJ))
            fj = next(s for s in kets if isinstance(s, AngularKetFJ))
            if is_unknown(fj.j_r) or is_unknown(fj.j_c) or is_unknown(jj.j_tot) or is_unknown(fj.f_c):
                raise RuntimeError("Cannot calculate overlap between JJ and FJ ket, due to Unknown quantum numbers.")
            return clebsch_gordan_6j(fj.j_r, fj.j_c, fj.i_c, jj.j_tot, fj.f_c, fj.f_tot)

        # JJ - LS overlaps
        if any(isinstance(s, AngularKetJJ) for s in kets) and any(isinstance(s, AngularKetLS) for s in kets):
            jj = next(s for s in kets if isinstance(s, AngularKetJJ))
            ls = next(s for s in kets if isinstance(s, AngularKetLS))
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
        if any(isinstance(s, AngularKetFJ) for s in kets) and any(isinstance(s, AngularKetLS) for s in kets):
            fj = next(s for s in kets if isinstance(s, AngularKetFJ))
            ls = next(s for s in kets if isinstance(s, AngularKetLS))
            ov: float = 0
            for coeff, jj_ket in fj.to_state("JJ"):
                ov += coeff * ls.calc_reduced_overlap(jj_ket)
            return float(ov)

        raise NotImplementedError(f"This method is not yet implemented for {self!r} and {other!r}.")

    def calc_reduced_matrix_element(  # noqa: C901, PLR0912
        self: Self, other: AngularKetBase[Any], operator: AngularOperatorType, kappa: int
    ) -> float:
        r"""Calculate the reduced angular matrix element.

        We follow equation (7.1.7) from Edmonds 1985 "Angular Momentum in Quantum Mechanics".
        This means, calculate the following matrix element:

        .. math::
            \left\langle self || \hat{O}^{(\kappa)} || other \right\rangle

        """
        if not is_angular_operator_type(operator):
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        if self.coupling_scheme != other.coupling_scheme:
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)
        if is_angular_momentum_quantum_number(operator) and operator not in self.quantum_number_names:
            return self.to_state().calc_reduced_matrix_element(other.to_state(), operator, kappa)

        qn_name: AngularMomentumQuantumNumbers
        if operator == "spherical":
            qn_name = "l_r"
        elif operator in self.quantum_number_names:
            if kappa != 1:
                raise ValueError("Only kappa=1 is supported for spin operators.")
            qn_name = operator
        elif operator.startswith("identity_"):
            if kappa != 0:
                raise ValueError("Only kappa=0 is supported for identity operator.")
            qn_name = operator.replace("identity_", "")  # type: ignore [assignment]
        else:
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        qn_self, qn_other = self.get_qn(qn_name), other.get_qn(qn_name)
        if is_unknown(qn_self) or is_unknown(qn_other):
            return 0  # TODO, ignore Unknown contributions for now

        if operator == "spherical":
            complete_reduced_matrix_element = calc_reduced_spherical_matrix_element(qn_self, qn_other, kappa)
        elif operator in self.quantum_number_names:
            complete_reduced_matrix_element = calc_reduced_spin_matrix_element(qn_self, qn_other)
        elif operator.startswith("identity_"):
            complete_reduced_matrix_element = calc_reduced_identity_matrix_element(qn_self, qn_other)
        else:
            raise NotImplementedError(f"calc_reduced_matrix_element is not implemented for operator {operator}.")

        if complete_reduced_matrix_element == 0:
            return 0
        if self._kronecker_delta_non_involved_spins(other, qn_name) == 0:
            return 0
        prefactor = self._calc_prefactor_of_operator_in_coupled_scheme(other, qn_name, kappa)
        return prefactor * complete_reduced_matrix_element

    def calc_matrix_element(
        self, other: AngularKetBase[Any], operator: AngularOperatorType, kappa: int, q: int
    ) -> float:
        r"""Calculate the dimensionless angular matrix element.

        Use the Wigner-Eckart theorem to calculate the angular matrix element from the reduced matrix element.
        We stick to the convention from Edmonds 1985 "Angular Momentum in Quantum Mechanics", see equation (5.4.1).
        This means, calculate the following matrix element:

        .. math::
            \left\langle self | \hat{O}^{(\kappa)}_q | other \right\rangle
            = <\alpha',f_{tot}',m'| \hat{O}^{(\kappa)}_q |\alpha,f_{tot},m>
            = (-1)^{(f_{tot} - m)} \cdot \mathrm{Wigner3j}(f_{tot}', \kappa, f_{tot}, -m', q, m)
                \cdot <\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>

        where alpha denotes all other quantum numbers
        and :math:`<\alpha',f_{tot}' || \hat{O}^{(\kappa)} || \alpha,f_{tot}>` is the reduced matrix element
        (see `calc_reduced_matrix_element`).

        Args:
            other: The other AngularKet :math:`|other>`.
            operator: The operator type :math:`\hat{O}_{kq}` for which to calculate the matrix element.
                E.g. 'spherical', 's_tot', 'l_r', etc.
            kappa: The rank :math:`\kappa` of the angular momentum operator.
            q: The component :math:`q` of the angular momentum operator.

        Returns:
            The dimensionless angular matrix element.

        """
        if is_not_set(self.m) or is_not_set(other.m):
            raise RuntimeError("m must be set to calculate the matrix element.")

        prefactor = self._calc_wigner_eckart_prefactor(other, kappa, q)
        reduced_matrix_element = self.calc_reduced_matrix_element(other, operator, kappa)
        return prefactor * reduced_matrix_element

    def _calc_wigner_eckart_prefactor(self, other: AngularKetBase[Any], kappa: int, q: int) -> float:
        if is_not_set(self.m) or is_not_set(other.m):
            raise RuntimeError("m must be set to calculate the Wigner-Eckart prefactor.")
        return minus_one_pow(self.f_tot - self.m) * calc_wigner_3j(self.f_tot, kappa, other.f_tot, -self.m, q, other.m)

    def _kronecker_delta_non_involved_spins(self, other: AngularKetBase[Any], qn: AngularMomentumQuantumNumbers) -> int:
        """Calculate the Kronecker delta for non involved angular momentum quantum numbers.

        This means return 0 if any of the quantum numbers,
        that are not qn or a coupled quantum number resulting from qn differ between self and other.
        """
        if qn not in self.quantum_number_names:
            raise ValueError(f"Quantum number {qn} is not a valid angular momentum quantum number for {self!r}.")

        resulting_qns = {qn}
        last_qn = qn
        while last_qn != "f_tot":
            for key, qs in self.coupled_quantum_numbers.items():
                if last_qn in qs:
                    resulting_qns.add(key)
                    last_qn = key
                    break
            else:
                raise ValueError(
                    f"_kronecker_delta_non_involved_spins: {last_qn} not found in coupled_quantum_numbers."
                )

        non_involved_qns = set(self.quantum_number_names) - resulting_qns
        for _qn in non_involved_qns:
            if self.get_qn(_qn) != other.get_qn(_qn):
                return 0
        return 1

    def _calc_prefactor_of_operator_in_coupled_scheme(
        self, other: AngularKetBase[Any], qn: AngularMomentumQuantumNumbers, kappa: int
    ) -> float:
        """Calculate the prefactor for the complete reduced matrix element.

        This approach is only valid if the operator acts only on one of the well defined quantum numbers.
        """
        if self.coupling_scheme != other.coupling_scheme:
            raise ValueError(
                "Both kets must be expressed in the same coupling scheme to calculate the prefactor of the operator."
            )

        if qn == "f_tot":
            return 1

        for key, qs in self.coupled_quantum_numbers.items():
            if qn not in qs:
                continue
            qn_combined = key
            # NOTE: the order does actually matter for the sign of some matrix elements
            # we use this to convention to stay consistent with the old pairinteraction database signs
            qn2, qn1 = qs
            operator_acts_on: Literal["first", "second"] = "first" if qn == qn1 else "second"
            break
        else:  # no break
            raise ValueError(f"Quantum number {qn} not found in coupled_quantum_numbers.")

        f1, f2, f_tot = (self.get_qn(qn1), self.get_qn(qn2), self.get_qn(qn_combined))
        i1, i2, i_tot = (other.get_qn(qn1), other.get_qn(qn2), other.get_qn(qn_combined))

        if (operator_acts_on == "first" and f2 != i2) or (operator_acts_on == "second" and f1 != i1):
            return 0
        if (
            is_unknown(f1)
            or is_unknown(f2)
            or is_unknown(f_tot)
            or is_unknown(i1)
            or is_unknown(i2)
            or is_unknown(i_tot)
        ):
            return 0  # TODO, ignore Unknown contributions for now
        prefactor = calc_prefactor_of_operator_in_coupled_scheme(f1, f2, f_tot, i1, i2, i_tot, kappa, operator_acts_on)
        return prefactor * self._calc_prefactor_of_operator_in_coupled_scheme(other, qn_combined, kappa)


class AngularKetLS(AngularKetBase[GenericT_Unknown], Generic[GenericT_Unknown]):
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

    s_tot: float | GenericT_Unknown
    """Total electron spin quantum number (s_c + s_r)."""
    l_tot: int | GenericT_Unknown
    """Total electron orbital quantum number (l_c + l_r)."""
    j_tot: float | GenericT_Unknown
    """Total electron angular momentum quantum number (s_tot + l_tot)."""

    @overload
    def __init__(
        self: AngularKetLS[AllKnown],
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self: AngularKetLS[Unknown],
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: Literal[True],
    ) -> None: ...

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: bool = False,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, name=name, species=species, allow_unknown=allow_unknown)

        self.s_tot = try_trivial_spin_addition(self.s_c, self.s_r, s_tot)  # type: ignore [assignment]
        self.l_tot = try_trivial_spin_addition(self.l_c, self.l_r, l_tot)  # type: ignore [assignment]
        if not is_unknown(self.l_tot):
            self.l_tot = int(self.l_tot)
        self.j_tot = try_trivial_spin_addition(self.l_tot, self.s_tot, j_tot)  # type: ignore [assignment]
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


class AngularKetJJ(AngularKetBase[GenericT_Unknown], Generic[GenericT_Unknown]):
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

    j_c: float | GenericT_Unknown
    """Total core electron angular quantum number (s_c + l_c)."""
    j_r: float | GenericT_Unknown
    """Total rydberg electron angular quantum number (s_r + l_r)."""
    j_tot: float | GenericT_Unknown
    """Total electron angular momentum quantum number (j_c + j_r)."""

    @overload
    def __init__(
        self: AngularKetJJ[AllKnown],
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self: AngularKetJJ[Unknown],
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: Literal[True],
    ) -> None: ...

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: bool = False,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, name=name, species=species, allow_unknown=allow_unknown)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c)  # type: ignore [assignment]
        self.j_r = try_trivial_spin_addition(self.l_r, self.s_r, j_r)  # type: ignore [assignment]
        self.j_tot = try_trivial_spin_addition(self.j_c, self.j_r, j_tot)  # type: ignore [assignment]
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


class AngularKetFJ(AngularKetBase[GenericT_Unknown], Generic[GenericT_Unknown]):
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

    j_c: float | GenericT_Unknown
    """Total core electron angular quantum number (s_c + l_c)."""
    f_c: float | GenericT_Unknown
    """Total core angular quantum number (j_c + i_c)."""
    j_r: float | GenericT_Unknown
    """Total rydberg electron angular quantum number (s_r + l_r)."""

    @overload
    def __init__(
        self: AngularKetFJ[AllKnown],
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self: AngularKetFJ[Unknown],
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: Literal[True],
    ) -> None: ...

    def __init__(
        self,
        i_c: float | None = None,
        s_c: float | None = None,
        l_c: int | Unknown | None = 0,
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
        allow_unknown: bool = False,
    ) -> None:
        """Initialize the Spin ket."""
        super().__init__(i_c, s_c, l_c, s_r, l_r, f_tot, m, name=name, species=species, allow_unknown=allow_unknown)

        self.j_c = try_trivial_spin_addition(self.l_c, self.s_c, j_c)  # type: ignore [assignment]
        self.j_r = try_trivial_spin_addition(self.l_r, self.s_r, j_r)  # type: ignore [assignment]
        self.f_c = try_trivial_spin_addition(self.j_c, self.i_c, f_c)  # type: ignore [assignment]
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
        name = None
        if is_unknown(self.l_c) or is_unknown(self.j_c) or is_unknown(self.f_c):
            name = ""
        return CoreKet(i_c=self.i_c, s_c=self.s_c, l_c=self.l_c, j_c=self.j_c, f_c=self.f_c, name=name)  # type: ignore [arg-type]
