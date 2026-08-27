from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rydstate.basis import BasisMQDT, BasisSQDT
    from rydstate.rydberg_state.rydberg_base import RydbergState


logger = logging.getLogger(__name__)

COLUMNS: dict[str, type] = {
    "id": int,
    "energy": float,
    "parity": int,
    "n": int,
    "nu": float,
    "f": float,
    "exp_nui": float,
    "exp_i_core": float,
    "exp_s_core": float,
    "exp_l_core": float,
    "exp_j_core": float,
    "exp_f_core": float,
    "exp_s_ryd": float,
    "exp_l_ryd": float,
    "exp_j_ryd": float,
    "exp_s": float,
    "exp_l": float,
    "exp_j": float,
    "std_nui": float,
    "std_i_core": float,
    "std_s_core": float,
    "std_l_core": float,
    "std_j_core": float,
    "std_f_core": float,
    "std_s_ryd": float,
    "std_l_ryd": float,
    "std_j_ryd": float,
    "std_s": float,
    "std_l": float,
    "std_j": float,
}


def generate_states_table(
    basis: BasisMQDT | BasisSQDT,
) -> dict[str, list[float | int | str | bool]]:
    """Calculate the states table for a given Basis."""
    basis.sort_states("nu")  # sort by nu == sort by energy

    states_data: list[tuple[float | int | str | bool, ...]] = []
    for ids, state in enumerate(basis.states):
        states_data.append(get_state_data(ids, state))

    assert len(states_data) == 0 or len(COLUMNS) == len(states_data[0])
    logger.info("Created the 'states' table (%s rows)", len(states_data))

    table = {column: [dtype(row[i]) for row in states_data] for i, (column, dtype) in enumerate(COLUMNS.items())}
    if np.any(np.diff(table["energy"]) < 0):
        raise ValueError("The energy of the states must be increasing with the id.")
    return table


def get_state_data(ids: int, state: RydbergState) -> tuple[float | int | str | bool, ...]:
    """Get the data for a given state as a tuple."""
    state_ls = state.to_coupling_scheme("LS")
    state_fj = state.to_coupling_scheme("FJ")

    data = (
        ids,  # id
        state.get_energy("a.u."),  # energy
        state.parity,  # parity = (-1)^(l_r + l_c)
        state.n,  # n
        state.nu,  # nu
        state.f_tot,  # f_tot
        state.calc_exp_qn("nui"),  # exp_nui
        state.calc_exp_qn("i_c"),  # exp_i_core
        state.calc_exp_qn("s_c"),  # exp_s_core
        state.calc_exp_qn("l_c"),  # exp_l_core
        state_fj.calc_exp_qn("j_c"),  # exp_j_core
        state_fj.calc_exp_qn("f_c"),  # exp_f_core
        state.calc_exp_qn("s_r"),  # exp_s_ryd
        state.calc_exp_qn("l_r"),  # exp_l_ryd
        state_fj.calc_exp_qn("j_r"),  # exp_j_ryd = j for sqdt only one valence electron
        state_ls.calc_exp_qn("s_tot"),  # exp_s
        state_ls.calc_exp_qn("l_tot"),  # exp_l
        state_ls.calc_exp_qn("j_tot"),  # exp_j
        state.calc_std_qn("nui"),  # std_nui = 0
        state.calc_std_qn("i_c"),  # std_i_core = 0
        state.calc_std_qn("s_c"),  # std_s_core = 0
        state.calc_std_qn("l_c"),  # std_l_core
        state_fj.calc_std_qn("j_c"),  # std_j_core
        state_fj.calc_std_qn("f_c"),  # std_f_core
        state.calc_std_qn("s_r"),  # std_s_ryd = 0
        state.calc_std_qn("l_r"),  # std_l_ryd
        state_fj.calc_std_qn("j_r"),  # std_j_ryd
        state_ls.calc_std_qn("s_tot"),  # std_s
        state_ls.calc_std_qn("l_tot"),  # std_l
        state_ls.calc_std_qn("j_tot"),  # std_j
    )
    return tuple(x.item() if isinstance(x, np.generic) else x for x in data)
