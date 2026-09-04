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

    table: dict[str, list[float | int | str | bool]] = {column: [] for column in COLUMNS}
    for ids, state in enumerate(basis.states):
        data = get_state_data(ids, state)
        for column, value in data.items():
            table[column].append(COLUMNS[column](value))
    assert all(len(values) == len(basis.states) for values in table.values()), "All columns must have the same length."

    logger.info("Created the 'states' table (%s rows)", len(basis.states))

    if np.any(np.diff(table["energy"]) < 0):
        raise ValueError("The energy of the states must be increasing with the id.")
    return table


def get_state_data(ids: int, state: RydbergState) -> dict[str, float | int | str | bool]:
    """Get the data for a given state as a dict, keyed by the column names."""
    state_ls = state.to_coupling_scheme("LS")
    state_fj = state.to_coupling_scheme("FJ")

    data: dict[str, float | int | str | bool] = {
        "id": ids,
        "energy": state.get_energy("a.u."),
        "parity": state.parity,  # parity = (-1)^(l_r + l_c)
        "n": state.n,
        "nu": state.nu,
        "f": state.f_tot,
        "exp_nui": state.calc_exp_qn("nui"),
        "exp_i_core": state.calc_exp_qn("i_c"),
        "exp_s_core": state.calc_exp_qn("s_c"),
        "exp_l_core": state.calc_exp_qn("l_c"),
        "exp_j_core": state_fj.calc_exp_qn("j_c"),
        "exp_f_core": state_fj.calc_exp_qn("f_c"),
        "exp_s_ryd": state.calc_exp_qn("s_r"),
        "exp_l_ryd": state.calc_exp_qn("l_r"),
        "exp_j_ryd": state_fj.calc_exp_qn("j_r"),
        "exp_s": state_ls.calc_exp_qn("s_tot"),
        "exp_l": state_ls.calc_exp_qn("l_tot"),
        "exp_j": state_ls.calc_exp_qn("j_tot"),
        "std_nui": state.calc_std_qn("nui"),
        "std_i_core": state.calc_std_qn("i_c"),
        "std_s_core": state.calc_std_qn("s_c"),
        "std_l_core": state.calc_std_qn("l_c"),
        "std_j_core": state_fj.calc_std_qn("j_c"),
        "std_f_core": state_fj.calc_std_qn("f_c"),
        "std_s_ryd": state.calc_std_qn("s_r"),
        "std_l_ryd": state.calc_std_qn("l_r"),
        "std_j_ryd": state_fj.calc_std_qn("j_r"),
        "std_s": state_ls.calc_std_qn("s_tot"),
        "std_l": state_ls.calc_std_qn("l_tot"),
        "std_j": state_ls.calc_std_qn("j_tot"),
    }
    return {key: value.item() if isinstance(value, np.generic) else value for key, value in data.items()}
