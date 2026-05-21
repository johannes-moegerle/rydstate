from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rydstate.angular.angular_ket import AngularKetLS

if TYPE_CHECKING:
    import sqlite3

    from rydstate.basis import BasisSQDT
    from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDT


logger = logging.getLogger(__name__)

COLUMNS = [
    "id",
    "energy",
    "parity",
    "n",
    "nu",
    "f",
    "exp_nui",
    "exp_l",
    "exp_j",
    "exp_s",
    "exp_l_ryd",
    "exp_j_ryd",
    "std_nui",
    "std_l",
    "std_j",
    "std_s",
    "std_l_ryd",
    "std_j_ryd",
    "is_j_total_momentum",
    "is_calculated_with_mqdt",
    "underspecified_channel_contribution",
]


def generate_states_table(
    basis: BasisSQDT,
    conn: sqlite3.Connection | None = None,
) -> list[tuple[float | int | str | bool, ...]]:
    """Populate the states table for a given species and n-range using BasisSQDT."""
    if basis.coupling_scheme != "LS":
        raise ValueError("Only LS coupling scheme is supported for now.")
    basis.sort_states("nu")  # sort by nu == sort by energy

    states_data: list[tuple[float | int | str | bool, ...]] = []
    for ids, state in enumerate(basis.states):
        states_data.append(get_state_data(ids, state))

    assert len(states_data) == 0 or len(COLUMNS) == len(states_data[0])

    if conn is not None:
        stmt = f"INSERT INTO states ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})"  # noqa: S608
        conn.executemany(stmt, states_data)
        num_rows = conn.execute("SELECT COUNT(*) FROM states").fetchone()[0]
        logger.info("Created the 'states' table (%s rows)", num_rows)

    return states_data


def get_state_data(ids: int, state: RydbergStateSQDT) -> tuple[float | int | str | bool, ...]:
    """Get the data for a given state as a tuple."""
    angular_ket = state.angular
    if not isinstance(angular_ket, AngularKetLS):
        raise TypeError("Only AngularKetLS is supported for now")

    angular_state = angular_ket.to_state()

    parity = -1 if angular_ket.l_tot % 2 == 1 else 1

    is_j_total_momentum = state.species.i_c == 0 or state.species.i_c is None
    is_calculated_with_mqdt = False

    return (
        ids,  # id
        state.species.get_ionization_energy("a.u.") + state.get_energy("a.u."),  # energy
        parity,  # parity = (-1)^l_tot
        state.n,  # n: quantum number
        state.nu,  # nu = NStar for sqdt
        angular_ket.f_tot,  # f: quantum number
        state.nu,  # exp_nui = nu for sqdt
        angular_ket.l_tot,  # exp_l = l
        angular_ket.j_tot,  # exp_j = j
        angular_ket.s_tot,  # exp_s = s
        angular_ket.l_r,  # exp_l_ryd = l for sqdt
        angular_state.calc_exp_qn("j_r"),  # exp_j_ryd = j for sqdt only one valence electron
        0,  # std_nui = 0
        0,  # std_l = 0
        0,  # std_j = 0
        0,  # std_s = 0
        0,  # std_l_ryd = 0
        angular_state.calc_std_qn("j_r"),  # std_j_ryd = 0 for sqdt and only one valence electron
        is_j_total_momentum,  # is_j_total_momentum = True for no hyperfine splitting
        is_calculated_with_mqdt,  # is_calculated_with_mqdt = False for sqdt
        0,  # underspecified_channel_contribution = 0 for sqdt
    )
