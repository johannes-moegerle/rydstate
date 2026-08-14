from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

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
    "exp_l": float,
    "exp_j": float,
    "exp_s": float,
    "exp_l_ryd": float,
    "exp_j_ryd": float,
    "std_nui": float,
    "std_l": float,
    "std_j": float,
    "std_s": float,
    "std_l_ryd": float,
    "std_j_ryd": float,
    "is_j_total_momentum": bool,
    "is_calculated_with_mqdt": bool,
    "underspecified_channel_contribution": float,
}


def generate_states_table(
    basis: BasisMQDT | BasisSQDT[Any],
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
    underspecified_channel_contribution = sum(abs(coeff) ** 2 for coeff, ket in state if ket.angular.contains_unknown)

    data = (
        ids,  # id
        state.get_energy("a.u."),  # energy
        state.parity,  # parity = (-1)^(l_r + l_c)
        state.n,  # n
        state.nu,  # nu
        state.f_tot,  # f_tot
        state.calc_exp_qn("nui"),  # exp_nui
        state.calc_exp_qn("l_tot"),  # exp_l
        state.calc_exp_qn("j_tot"),  # exp_j
        state.calc_exp_qn("s_tot"),  # exp_s
        state.calc_exp_qn("l_r"),  # exp_l_ryd
        state.calc_exp_qn("j_r"),  # exp_j_ryd = j for sqdt only one valence electron
        state.calc_std_qn("nui"),  # std_nui = 0
        state.calc_std_qn("l_tot"),  # std_l
        state.calc_std_qn("j_tot"),  # std_j
        state.calc_std_qn("s_tot"),  # std_s
        state.calc_std_qn("l_r"),  # std_l_ryd
        state.calc_std_qn("j_r"),  # std_j_ryd
        bool(state.element_properties.i_c == 0),  # is_j_total_momentum
        bool(len(state.rydberg_kets) > 1),  # is_calculated_with_mqdt
        underspecified_channel_contribution,  # underspecified_channel_contribution = 0 for sqdt
    )
    return tuple(x.item() if isinstance(x, np.generic) else x for x in data)
