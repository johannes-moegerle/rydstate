from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from rydstate.angular import AngularKetLS
from rydstate.radial.radial_ket import RadialKet
from rydstate.units import MatrixElementOperatorRanks

if TYPE_CHECKING:
    import sqlite3

    from rydstate.angular.utils import AngularOperatorType
    from rydstate.basis.basis_sqdt import BasisSQDT
    from rydstate.rydberg import RydbergStateSQDT
    from rydstate.units import MatrixElementOperator

logger = logging.getLogger(__name__)

COLUMNS = [
    "id_initial",
    "id_final",
    "val",
]

MATRIX_ELEMENTS_OF_INTEREST: dict[str, MatrixElementOperator] = {
    "matrix_elements_d": "electric_dipole",
    "matrix_elements_q": "electric_quadrupole",
    "matrix_elements_o": "electric_octupole",
    "matrix_elements_q0": "electric_quadrupole_zero",
    "matrix_elements_mu": "magnetic_dipole",
}


def generate_matrix_elements_tables(
    basis: BasisSQDT,
    conn: sqlite3.Connection | None = None,
    max_delta_n: float = np.inf,
    all_n_up_to: float = np.inf,
    k_angular_max: int = 3,
) -> dict[str, list[tuple[int, int, float]]]:
    """Populate matrix element tables for all relevant pairs of states."""
    basis.sort_states("nu")  # sort by nu == sort by energy
    list_of_id_state = list(enumerate(basis.states))
    list_of_id_state = sorted(list_of_id_state, key=lambda x: (x[1].angular.l_r, x[1].n, x[0]))

    matrix_elements: dict[str, list[tuple[int, int, float]]] = {tkey: [] for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for i, (id1, state1) in enumerate(list_of_id_state):
        list_filtered = filter(lambda x: x[1].angular.l_r - state1.angular.l_r <= k_angular_max, list_of_id_state[i:])
        for id2, state2 in list_filtered:
            if all(n > all_n_up_to for n in [state1.n, state2.n]) and abs(state1.n - state2.n) > max_delta_n:
                # If delta_n is larger than max_delta_n, we dont calculate the matrix elements anymore,
                # since these are so small, that they are usually not relevant for further calculations
                # However, we keep all dipole interactions with small n (we choose all_n_up_to as a cutoff)
                # since these are relevant for the spontaneous decay rates
                continue

            id_tuple = (id1, id2) if id1 <= id2 else (id2, id1)
            states = (state1, state2) if id1 <= id2 else (state2, state1)

            me_one_pair = calc_matrix_elements_one_pair(states[0], states[1], MATRIX_ELEMENTS_OF_INTEREST)
            for tkey, me in me_one_pair.items():
                matrix_elements[tkey].append((id_tuple[0], id_tuple[1], me))

            if id1 != id2:
                me_one_pair = calc_matrix_elements_one_pair(states[1], states[0], MATRIX_ELEMENTS_OF_INTEREST)
                for tkey, me in me_one_pair.items():
                    matrix_elements[tkey].append((id_tuple[1], id_tuple[0], me))

    for key, mes in matrix_elements.items():
        matrix_elements[key] = sorted(mes)
        assert len(mes) == 0 or len(COLUMNS) == len(mes[0])

    if conn is not None:
        for tkey, mes in matrix_elements.items():
            stmt = f"INSERT INTO {tkey} ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})"  # noqa: S608
            conn.executemany(stmt, mes)
            num_rows = conn.execute(f"SELECT COUNT(*) FROM {tkey}").fetchone()[0]  # noqa: S608
            logger.info("Created the '%s' table (%s rows)", tkey, num_rows)

    return matrix_elements


def calc_matrix_elements_one_pair(
    state1: RydbergStateSQDT,
    state2: RydbergStateSQDT,
    matrix_elements_of_interest: dict[str, MatrixElementOperator],
) -> dict[str, float]:
    matrix_elements: dict[str, float] = {}
    for tkey, operator in matrix_elements_of_interest.items():
        k_radial, k_angular = MatrixElementOperatorRanks[operator]

        if operator == "magnetic_dipole":
            # Magnetic dipole operator: mu = - mu_B (g_l <l_tot> + g_s <s_tot>)
            g_s = 2.0023192
            value_s_tot = calc_reduced_angular_matrix_element_cached(
                state1.angular.quantum_numbers, state2.angular.quantum_numbers, "s_tot", k_angular
            )
            g_l = 1
            value_l_tot = calc_reduced_angular_matrix_element_cached(
                state1.angular.quantum_numbers, state2.angular.quantum_numbers, "l_tot", k_angular
            )
            angular_matrix_element = g_s * value_s_tot + g_l * value_l_tot
            prefactor = -0.5  # - mu_B in atomic units

        elif operator in ["electric_dipole", "electric_quadrupole", "electric_octupole", "electric_quadrupole_zero"]:
            angular_matrix_element = calc_reduced_angular_matrix_element_cached(
                state1.angular.quantum_numbers, state2.angular.quantum_numbers, "spherical", k_angular
            )
            prefactor = math.sqrt(4 * math.pi / (2 * k_angular + 1))  # e in atomic units is 1
        else:
            raise NotImplementedError(f"Operator {operator} not implemented.")

        if angular_matrix_element == 0:
            continue

        radial_matrix_element_au = calc_radial_matrix_element_cached(
            state1.species.name,
            *(state1.n, state1.nu, state1.angular.l_r),
            *(state2.n, state2.nu, state2.angular.l_r),
            k_radial,
        )
        if radial_matrix_element_au == 0:
            continue

        matrix_elements[tkey] = prefactor * radial_matrix_element_au * angular_matrix_element

    return matrix_elements


@lru_cache(maxsize=100_000)
def calc_reduced_angular_matrix_element_cached(
    qns1: tuple[float, ...],
    qns2: tuple[float, ...],
    operator: AngularOperatorType,
    k_angular: int,
) -> float:
    ket1 = AngularKetLS(*qns1)  # type: ignore[arg-type]
    ket2 = AngularKetLS(*qns2)  # type: ignore[arg-type]
    # ket2 is the final state and ket1 the initial state
    # ket2.calc_reduced_matrix_element(ket1, T, k) gives the reduced matrix element <ket2||T^k||ket1>
    return ket2.calc_reduced_matrix_element(ket1, operator, k_angular)


def calc_radial_matrix_element_cached(
    species_name: str, n1: int, nu1: float, l1: int, n2: int, nu2: float, l2: int, k_radial: int
) -> float:
    if k_radial == 0 and nu1 == nu2:
        return 1 if l1 == l2 else 0

    if (nu1, l1) > (nu2, l2):  # for better use of the cache and since the radial matrix element is symmetric
        return _calc_radial_matrix_element_cached(species_name, n2, nu2, l2, n1, nu1, l1, k_radial)

    return _calc_radial_matrix_element_cached(species_name, n1, nu1, l1, n2, nu2, l2, k_radial)


# Cache size should be at least on the order of 4 * (all_n_up_to + 2 * max_delta_n)
# however, for the first n until n=all_n_up_to we need an even larger cache size
@lru_cache(maxsize=50_000)
def _calc_radial_matrix_element_cached(
    species_name: str, n1: int, nu1: float, l1: int, n2: int, nu2: float, l2: int, k_radial: int
) -> float:
    state1 = get_radial_state_cached(species_name, n1, nu1, l1)
    state2 = get_radial_state_cached(species_name, n2, nu2, l2)
    # state2 is the final state and state1 the initial state
    return state2.calc_matrix_element(state1, k_radial, unit="a.u.")


# Cache size should be one the order of N_MAX * 4 * 2
# (since for each initial state we loop over all l' = l, l+1, l+2 and l+3 final states (and all j final))
@lru_cache(maxsize=2_000)
def get_radial_state_cached(species_name: str, n: int, nu: float, l: int) -> RadialKet:
    """Get the cached rydberg state (where the wavefunction was already calculated)."""
    state = RadialKet(species_name, nu, l)
    state.set_n_for_sanity_check(n)
    state.create_wavefunction(sign_convention="n_l_1")
    return state
