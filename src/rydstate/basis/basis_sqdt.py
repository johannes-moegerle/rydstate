from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, overload

import numpy as np

from rydstate.angular import NotSet
from rydstate.angular.angular_ket import AngularKetBase, AngularKetFJ, AngularKetJJ, AngularKetLS
from rydstate.basis.basis_base import BasisBase
from rydstate.basis.utils import get_m_range, is_allowed_qn
from rydstate.rydberg_state import RydbergStateSQDT
from rydstate.species import get_sqdt
from rydstate.species.potential import Potential, get_potential_class
from rydstate.species.sqdt import SQDT

if TYPE_CHECKING:
    from rydstate.angular.utils import AllKnown, CouplingScheme

T_AngularKet = TypeVar("T_AngularKet", bound=AngularKetBase[Any])

logger = logging.getLogger(__name__)


class BasisSQDT(BasisBase[RydbergStateSQDT[T_AngularKet]], Generic[T_AngularKet]):
    states: list[RydbergStateSQDT[T_AngularKet]]

    @overload
    def __init__(
        self: BasisSQDT[AngularKetLS[AllKnown]],
        species: str,
        n: tuple[int, int],
        *,
        f_tot: tuple[float, float] | None = None,
        l_r: tuple[int, int] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        coupling_scheme: Literal["LS"] = "LS",
        sqdt: SQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: BasisSQDT[AngularKetJJ[AllKnown]],
        species: str,
        n: tuple[int, int],
        *,
        f_tot: tuple[float, float] | None = None,
        l_r: tuple[int, int] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        coupling_scheme: Literal["JJ"],
        sqdt: SQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: BasisSQDT[AngularKetFJ[AllKnown]],
        species: str,
        n: tuple[int, int],
        *,
        f_tot: tuple[float, float] | None = None,
        l_r: tuple[int, int] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        coupling_scheme: Literal["FJ"],
        sqdt: SQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None: ...

    def __init__(
        self,
        species: str,
        n: tuple[int, int],
        *,
        l_r: tuple[int, int] | None = None,
        f_tot: tuple[float, float] | None = None,
        m: tuple[float, float] | None | NotSet = NotSet,
        coupling_scheme: CouplingScheme = "LS",
        # potential and sqdt parameters
        sqdt: SQDT | str | None = None,
        potential_class: type[Potential] | str | None = None,
    ) -> None:
        """Initialize the SQDT basis.

        Args:
            species: Atomic species.
            n: Tuple of (n_min, n_max) for the principal quantum number.
            l_r: Optional tuple of (l_r_min, l_r_max) for the Rydberg electron orbital angular momentum.
                Default None, include all l_r values.
            f_tot: Optional tuple of (f_tot_min, f_tot_max) for the total angular momentum.
                Default None, include all f_tot values.
            m: Optional tuple of (m_min, m_max) for the magnetic quantum number.
                If None, all m values are included.
                Default NotSet, m is not specified and will be set to NotSet for all states.
            coupling_scheme: The coupling scheme to use for the angular kets.
            sqdt: The SQDT data to use for the states.
                Either an instance of an SQDT class
                or a string representing the tag of the SQDT class to use.
            potential_class: The potential class to use for the radial ket.
                Either a a potential class
                or a string representing the tag of the potential class to use.

        """
        super().__init__(species)
        self.sqdt = sqdt if isinstance(sqdt, SQDT) else get_sqdt(species, tag=sqdt)

        if isinstance(potential_class, type) and issubclass(potential_class, Potential):
            self.potential_class = potential_class
        else:
            self.potential_class = get_potential_class(species, tag=potential_class)

        self._init_states(n, f_tot, l_r, m, coupling_scheme)

    def _init_states(
        self,
        n_range: tuple[int, int],
        f_tot_range: tuple[float, float] | None,
        l_r_range: tuple[int, int] | None,
        m_range: tuple[float, float] | None | NotSet,
        coupling_scheme: CouplingScheme,
    ) -> None:
        self.coupling_scheme = coupling_scheme
        self.states = []

        if coupling_scheme == "LS":
            add_states = self._add_states_ls
        elif coupling_scheme == "JJ":
            add_states = self._add_states_jj
        elif coupling_scheme == "FJ":
            add_states = self._add_states_fj
        else:
            raise ValueError(f"Invalid coupling scheme: {coupling_scheme}")

        for n in range(n_range[0], n_range[1] + 1):
            for l_r in range(n):
                if not is_allowed_qn(l_r_range, l_r):
                    continue
                add_states(n, l_r, f_tot_range, m_range)
        self.states.sort(key=lambda state: state.nu)

    def _add_states_ls(
        self,
        n: int,
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        for s_tot in s_tot_list:
            if not self.element_properties.is_allowed_shell(n, l_r, s_tot):
                continue
            for j_tot in np.arange(abs(l_r - s_tot), l_r + s_tot + 1):
                for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                    if not is_allowed_qn(f_tot_range, f_tot):
                        continue
                    for m in get_m_range(f_tot, m_range):
                        angular_ket = AngularKetLS(
                            l_r=l_r, s_tot=s_tot, j_tot=j_tot, f_tot=f_tot, m=m, species=self.species
                        )
                        state = RydbergStateSQDT(
                            self.species,
                            n=n,
                            angular_ket=angular_ket,
                            sqdt=self.sqdt,
                            potential=self.potential_class(l_r),
                        )
                        self.states.append(state)  # type: ignore [arg-type]

    def _add_states_jj(
        self,
        n: int,
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = j_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        allowed = [self.element_properties.is_allowed_shell(n, l_r, s) for s in s_tot_list]
        if not all(allowed):
            if any(allowed):
                logger.warning(
                    "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                    "In JJ coupling the state does not exist, thus skipping this shell",
                    *(l_r, n),
                )
            return

        for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
            for j_tot in range(int(abs(j_r - j_c)), int(j_r + j_c + 1)):
                for f_tot in np.arange(abs(j_tot - i_c), j_tot + i_c + 1):
                    if not is_allowed_qn(f_tot_range, f_tot):
                        continue
                    for m in get_m_range(f_tot, m_range):
                        angular_ket = AngularKetJJ(
                            l_r=l_r, j_r=j_r, j_tot=j_tot, f_tot=f_tot, m=m, species=self.species
                        )
                        state = RydbergStateSQDT(
                            self.species,
                            n=n,
                            angular_ket=angular_ket,
                            sqdt=self.sqdt,
                            potential=self.potential_class(l_r),
                        )
                        self.states.append(state)  # type: ignore [arg-type]

    def _add_states_fj(
        self,
        n: int,
        l_r: int,
        f_tot_range: tuple[float, float] | None,
        m_range: tuple[float, float] | None | NotSet = NotSet,
    ) -> None:
        i_c = self.element_properties.i_c
        s_r = 0.5
        s_c = j_c = self.element_properties.s_c
        s_tot_list = np.arange(s_r - s_c, s_r + s_c + 1)

        allowed = [self.element_properties.is_allowed_shell(n, l_r, s) for s in s_tot_list]
        if not all(allowed):
            if any(allowed):
                logger.warning(
                    "For l=%d, n=%d one of the singlet/triplet states is not allowed. "
                    "In FJ coupling the state does not exist, thus skipping this shell",
                    *(l_r, n),
                )
            return

        for j_r in np.arange(abs(l_r - s_r), l_r + s_r + 1):
            for f_c in np.arange(abs(j_c - i_c), j_c + i_c + 1):
                for f_tot in np.arange(abs(f_c - j_r), f_c + j_r + 1):
                    if not is_allowed_qn(f_tot_range, f_tot):
                        continue
                    for m in get_m_range(f_tot, m_range):
                        angular_ket = AngularKetFJ(l_r=l_r, j_r=j_r, f_c=f_c, f_tot=f_tot, m=m, species=self.species)
                        state = RydbergStateSQDT(
                            self.species,
                            n=n,
                            angular_ket=angular_ket,
                            sqdt=self.sqdt,
                            potential=self.potential_class(l_r),
                        )
                        self.states.append(state)  # type: ignore [arg-type]
