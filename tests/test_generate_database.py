from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import RydbergStateSQDTAlkali
from rydstate.basis.basis_sqdt import BasisSQDT
from rydstate.generate_database.generate_database import DATABASE_SQL_FILE
from rydstate.generate_database.generate_matrix_elements_table import generate_matrix_elements_tables
from rydstate.generate_database.generate_misc_table import generate_wigner_table
from rydstate.generate_database.generate_states_table import generate_states_table, get_state_data

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def conn() -> Generator[sqlite3.Connection, None, None]:
    connection = sqlite3.connect(":memory:")
    connection.executescript(DATABASE_SQL_FILE.read_text(encoding="utf-8"))
    try:
        yield connection
    finally:
        connection.close()


def test_generate_wigner_table_returns_and_inserts_rows(conn: sqlite3.Connection) -> None:
    rows = generate_wigner_table(f_max=0, kappa_max=0, conn=conn)

    assert rows == [(0.0, 0.0, 0.0, 0.0, 0, 0, 1.0)]
    assert conn.execute("SELECT * FROM wigner").fetchall() == rows


def test_get_state_data_for_sqdt_alkali_state() -> None:
    state = RydbergStateSQDTAlkali("H", n=1, l=0, j=0.5)

    row = get_state_data(7, state)

    assert row[0] == 7
    assert row[2:6] == (1, 1, 1.0, 0.5)
    assert row[6:12] == (1.0, 0, 0.5, 0.5, 0, 0.5)
    assert row[12:] == (0, 0, 0, 0, 0, 0, True, False, 0)


def test_generate_states_table_sorts_returns_and_inserts_rows(conn: sqlite3.Connection) -> None:
    basis = BasisSQDT("H", n=(1, 2), coupling_scheme="LS")
    basis.filter_states("l_r", 0)

    rows = generate_states_table(basis, conn=conn)

    assert [row[0] for row in rows] == [0, 1]
    assert [row[3] for row in rows] == [1, 2]
    assert conn.execute("SELECT id, n FROM states ORDER BY id").fetchall() == [(0, 1), (1, 2)]


def test_generate_matrix_elements_tables(
    conn: sqlite3.Connection,
) -> None:
    basis = BasisSQDT("H", n=(10, 10), coupling_scheme="LS")
    basis.filter_states("l_r", (0, 2))
    # only keep states with j = l + 1/2 for easier testing
    basis.states = [state for state in basis.states if state.angular.get_qn("j_tot") == state.angular.l_r + 0.5]
    basis.sort_states("nu")
    states = basis.states

    for state in states:
        state.radial.create_wavefunction(sign_convention="n_l_1")

    rows_by_table = generate_matrix_elements_tables(basis, conn=conn)
    reference_table = [
        (0, 1, states[1].calc_reduced_matrix_element(states[0], "electric_dipole", unit="a.u.")),
        (1, 0, states[0].calc_reduced_matrix_element(states[1], "electric_dipole", unit="a.u.")),
        (1, 2, states[2].calc_reduced_matrix_element(states[1], "electric_dipole", unit="a.u.")),
        (2, 1, states[1].calc_reduced_matrix_element(states[2], "electric_dipole", unit="a.u.")),
    ]

    assert len(rows_by_table["matrix_elements_d"]) == len(reference_table)
    for i, row in enumerate(rows_by_table["matrix_elements_d"]):
        assert row[:2] == reference_table[i][:2]
        assert np.isclose(row[2], reference_table[i][2])
