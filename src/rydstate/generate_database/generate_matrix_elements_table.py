from __future__ import annotations

import logging
from bisect import bisect_right
from typing import TYPE_CHECKING, Any

from rydstate.angular.utils import is_unknown
from rydstate.units import MatrixElementOperatorRanks

if TYPE_CHECKING:
    from rydstate.basis import BasisMQDT, BasisSQDT
    from rydstate.rydberg_state.rydberg_base import RydbergStateBase
    from rydstate.units import MatrixElementOperator

logger = logging.getLogger(__name__)

COLUMNS: dict[str, type] = {
    "id_initial": int,
    "id_final": int,
    "val": float,
}

MATRIX_ELEMENTS_OF_INTEREST: dict[str, MatrixElementOperator] = {
    "matrix_elements_d": "electric_dipole",
    "matrix_elements_q": "electric_quadrupole",
    "matrix_elements_o": "electric_octupole",
    "matrix_elements_q0": "electric_quadrupole_zero",
    "matrix_elements_mu": "magnetic_dipole",
}


def generate_matrix_elements_tables(
    basis: BasisMQDT | BasisSQDT[Any],
    max_delta_nu: float = float("inf"),
    all_nu_up_to: float = float("inf"),
    *,
    free_memory: bool = False,
) -> dict[str, dict[str, list[int | float]]]:
    """Calculate matrix element tables for all relevant pairs of states."""
    k_angular_max = max(MatrixElementOperatorRanks[op][1] for op in MATRIX_ELEMENTS_OF_INTEREST.values())

    basis.sort_states("nu")  # sort by nu == sort by energy
    list_of_id_state = list(enumerate(basis.states))

    # precomupte l_r values for efficient k_angular_max filtering
    # (channels with unknown l_r are ignored here, they never contribute to any of the matrix elements of interest)
    l_r_sets = [
        {ket.angular.l_r for ket in state.rydberg_kets if not is_unknown(ket.angular.l_r)}
        for _, state in list_of_id_state
    ]
    # sort the states by their smallest l_r (and nu and id)
    sort_order = sorted(
        range(len(list_of_id_state)),
        key=lambda i: (min(l_r_sets[i]), list_of_id_state[i][1].nu, list_of_id_state[i][0]),
    )
    list_of_id_state = [list_of_id_state[i] for i in sort_order]
    l_r_min = [min(l_r_sets[i]) for i in sort_order]
    l_r_max = [max(l_r_sets[i]) for i in sort_order]
    assert sorted(l_r_min) == l_r_min, "l_r_min is not sorted"

    matrix_elements: dict[str, list[tuple[int, int, float]]] = {tkey: [] for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for i1, (id1, state1) in enumerate(list_of_id_state):
        # Because l_r_min is sorted, for all states from i2_stop on, every channel differs by more than k_angular_max
        # in l_r from every channel of state1, so all their matrix elements with state1 vanish, and we can skip them.
        i2_stop = bisect_right(l_r_min, l_r_max[i1] + k_angular_max)
        nu1_above_cutoff = state1.nu > all_nu_up_to
        for id2, state2 in list_of_id_state[i1:i2_stop]:
            if nu1_above_cutoff and state2.nu > all_nu_up_to and abs(state1.nu - state2.nu) > max_delta_nu + 0.5:
                # If delta_nu is larger than max_delta_nu (+0.5 to not lose states compared to previous max_delta_n)
                # we dont calculate the matrix elements anymore,
                # since these are so small, that they are usually not relevant for further calculations
                # However, we keep all dipole interactions with small n (we choose all_nu_up_to as a cutoff)
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

        if free_memory:
            state1.free_memory()

    tables: dict[str, dict[str, list[int | float]]] = {}
    for tkey, mes in matrix_elements.items():
        # sort such that (i, j) is directly followed by (j, i); their values are identical up to the sign,
        # so keeping them adjacent roughly halves the parquet file size after compression
        mes_sorted = sorted(mes, key=lambda row: (min(row[0], row[1]), max(row[0], row[1]), row[0]))
        assert len(mes_sorted) == 0 or len(COLUMNS) == len(mes_sorted[0])
        tables[tkey] = {
            column: [dtype(row[i]) for row in mes_sorted] for i, (column, dtype) in enumerate(COLUMNS.items())
        }
        logger.info("Created the '%s' table (%s rows)", tkey, len(mes_sorted))

    return tables


def calc_matrix_elements_one_pair(
    initial: RydbergStateBase, final: RydbergStateBase, matrix_elements_of_interest: dict[str, MatrixElementOperator]
) -> dict[str, float]:
    matrix_elements: dict[str, float] = {}
    for tkey, operator in matrix_elements_of_interest.items():
        me = final.calc_reduced_matrix_element(initial, operator, unit="a.u.")
        if me != 0:
            matrix_elements[tkey] = me
    return matrix_elements
