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


TEST_SPECIES = ["H", "Li", "Na", "K", "Rb", "Cs", "Sr88_sqdt", "Yb174_sqdt"]


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


@pytest.mark.parametrize("species", TEST_SPECIES)
def test_generate_states_table(species: str, conn: sqlite3.Connection) -> None:
    basis = BasisSQDT(species, n=(20, 21), coupling_scheme="LS")
    basis.filter_states("l_r", (0, 2))
    basis.sort_states("nu")

    rows = generate_states_table(basis, conn=conn)

    assert len(basis.states) > 2
    assert len(rows) == len(basis.states)

    data = np.array(conn.execute("SELECT n, exp_l_ryd, exp_s FROM states").fetchall())
    assert np.allclose(data[:, 0], basis.calc_exp_qn("n"))
    assert np.allclose(data[:, 1], basis.calc_exp_qn("l_r"))
    assert np.allclose(data[:, 2], basis.calc_exp_qn("s_tot"))


@pytest.mark.parametrize("species", TEST_SPECIES)
def test_generate_matrix_elements_table(species: str, conn: sqlite3.Connection) -> None:
    basis = BasisSQDT(species, n=(20, 21), coupling_scheme="LS")
    basis.filter_states("l_r", (0, 2))
    basis.sort_states("nu")

    rows_by_table = generate_matrix_elements_tables(basis, conn=conn)

    for rows in rows_by_table.values():
        assert len(rows) > 2

    states = basis.states
    for state in states:
        state.radial.create_wavefunction(sign_convention="n_l_1")
    for row in rows_by_table["matrix_elements_d"]:
        reference = states[row[1]].calc_reduced_matrix_element(states[row[0]], "electric_dipole", unit="a.u.")
        assert np.isclose(row[2], reference)
