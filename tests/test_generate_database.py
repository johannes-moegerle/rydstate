from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from rydstate import RydbergStateSQDTAlkali
from rydstate.basis import BasisMQDT, BasisSQDT
from rydstate.generate_database.generate_matrix_elements_table import (
    MATRIX_ELEMENTS_OF_INTEREST,
    calc_matrix_elements_one_pair,
    generate_matrix_elements_tables,
)
from rydstate.generate_database.generate_misc_table import generate_wigner_table
from rydstate.generate_database.generate_states_table import generate_states_table, get_state_data
from rydstate.units import MatrixElementOperatorRanks

TEST_SPECIES_SPECIFIER = [
    *["H", "Li", "Na", "K", "Rb", "Cs"],
    *["Sr88_sqdt", "Yb174_sqdt"],
    *["Sr87_mqdt", "Sr88_mqdt", "Yb171_mqdt", "Yb173_mqdt", "Yb174_mqdt"],
]


def test_generate_wigner_table_returns_rows() -> None:
    table = generate_wigner_table(f_max=0, kappa_max=0)

    assert table == {
        "f_initial": [0.0],
        "f_final": [0.0],
        "m_initial": [0.0],
        "m_final": [0.0],
        "kappa": [0],
        "q": [0],
        "val": [1.0],
    }


def test_get_state_data_for_sqdt_alkali_state() -> None:
    state = RydbergStateSQDTAlkali("H", n=1, l=0, j=0.5)
    row = get_state_data(7, state)
    assert row[0] == 7
    assert row[2:6] == (1, 1, 1.0, 0.5)
    assert row[6:12] == (1.0, 0, 0.5, 0.5, 0, 0.5)
    assert row[12:] == (0, 0, 0, 0, 0, 0, True, False, 0)


@pytest.mark.parametrize("species_specifier", TEST_SPECIES_SPECIFIER)
def test_generate_states_table(species_specifier: str) -> None:
    species = species_specifier.removesuffix("_mqdt").removesuffix("_sqdt")
    basis: BasisMQDT | BasisSQDT[Any]
    if species_specifier.endswith("_mqdt"):
        basis = BasisMQDT(species, nu=(50, 52), l_r=(0, 2))
    else:
        basis = BasisSQDT(species, n=(50, 52), l_r=(0, 2), coupling_scheme="LS")
    basis.sort_states("nu")

    table = generate_states_table(basis)

    assert len(basis.states) > 2
    assert all(len(values) == len(basis.states) for values in table.values())

    assert np.allclose(table["nu"], basis.calc_exp_qn("nu"))
    assert np.allclose(table["exp_l_ryd"], basis.calc_exp_qn("l_r"))
    assert np.allclose(table["exp_s"], basis.calc_exp_qn("s_tot"))


@pytest.mark.parametrize("species_specifier", TEST_SPECIES_SPECIFIER)
def test_generate_matrix_elements_table(species_specifier: str) -> None:
    species = species_specifier.removesuffix("_mqdt").removesuffix("_sqdt")
    basis: BasisMQDT | BasisSQDT[Any]
    if species_specifier.endswith("_mqdt"):
        basis = BasisMQDT(species, nu=(50, 52), l_r=(0, 2))
    else:
        basis = BasisSQDT(species, n=(50, 52), l_r=(0, 2), coupling_scheme="LS")
    basis.sort_states("nu")

    tables = generate_matrix_elements_tables(basis, free_memory=False)

    for table in tables.values():
        assert all(len(values) > 2 for values in table.values())

    states = basis.states
    table = tables["matrix_elements_d"]
    for id_initial, id_final, val in zip(table["id_initial"], table["id_final"], table["val"], strict=True):
        reference = states[int(id_final)].calc_reduced_matrix_element(
            states[int(id_initial)], "electric_dipole", unit="a.u."
        )
        assert np.isclose(val, reference)


@pytest.mark.parametrize("species_specifier", ["Yb174_mqdt", "Rb"])
def test_matrix_elements_table_matches_unfiltered_reference(species_specifier: str) -> None:
    """Compare the matrix element tables against an unfiltered loop over all pairs of states.

    generate_matrix_elements_tables only evaluates the state pairs that survive the l_r window
    (bisect on the sorted l_r_min) and the f_tot triangle rule. Both filters can only fail by
    silently omitting rows, which test_generate_matrix_elements_table cannot detect since it only
    checks the values that are present.
    Comparing the values as well also covers the (j, i) rows, which are not calculated but derived
    from the (i, j) rows via the symmetry of the reduced matrix elements.
    """
    species = species_specifier.removesuffix("_mqdt").removesuffix("_sqdt")
    basis: BasisMQDT | BasisSQDT[Any]
    if species_specifier.endswith("_mqdt"):
        # Yb174 also has channels with unknown l_r, which the l_r filter assumes to never contribute
        basis = BasisMQDT(species, nu=(50, 52), l_r=(0, 5))
    else:
        basis = BasisSQDT(species, n=(50, 51), l_r=(0, 7), coupling_scheme="LS")
    basis.sort_states("nu")

    # make sure the basis actually contains pairs that the filters skip, otherwise the test is vacuous
    k_angular_max = max(MatrixElementOperatorRanks[op][1] for op in MATRIX_ELEMENTS_OF_INTEREST.values())
    exp_l_r = basis.calc_exp_qn("l_r")
    f_tot = [state.f_tot for state in basis.states]
    assert max(exp_l_r) - min(exp_l_r) > k_angular_max + 1, "basis does not exercise the l_r filter"
    assert max(f_tot) - min(f_tot) > k_angular_max + 1, "basis does not exercise the f_tot filter"

    tables = generate_matrix_elements_tables(basis, free_memory=False)

    expected: dict[str, dict[tuple[int, int], float]] = {tkey: {} for tkey in MATRIX_ELEMENTS_OF_INTEREST}
    for id_initial, initial in enumerate(basis.states):
        for id_final, final in enumerate(basis.states):
            for tkey, me in calc_matrix_elements_one_pair(initial, final, MATRIX_ELEMENTS_OF_INTEREST).items():
                expected[tkey][id_initial, id_final] = me

    for tkey, table in tables.items():
        pairs = list(zip(table["id_initial"].tolist(), table["id_final"].tolist(), strict=True))
        assert len(pairs) == len(set(pairs)), f"{tkey}: table contains duplicate rows"

        missing = sorted(expected[tkey].keys() - set(pairs))
        extra = sorted(set(pairs) - expected[tkey].keys())
        assert not missing, f"{tkey}: {len(missing)} non-vanishing matrix elements were filtered out: {missing[:5]}"
        assert not extra, f"{tkey}: {len(extra)} rows for vanishing matrix elements: {extra[:5]}"

        reference = [expected[tkey][pair] for pair in pairs]
        np.testing.assert_allclose(table["val"], reference, rtol=1e-10, atol=0, err_msg=f"wrong values in '{tkey}'")
