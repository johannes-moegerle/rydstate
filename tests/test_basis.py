import numpy as np
import pytest
from rydstate import BasisSQDT


@pytest.mark.parametrize("species_name", ["Rb", "Na", "H"])
def test_alkali_basis(species_name: str) -> None:
    """Test alkali basis creation."""
    basis = BasisSQDT(species_name, n=(1, 20), coupling_scheme="LS")
    basis.sort_states("n", "l_r")
    lowest_n_state = {"Rb": (4, 2), "Na": (3, 0), "H": (1, 0)}[species_name]
    assert (basis.states[0].n, basis.states[0].angular.l_r) == lowest_n_state
    assert (basis.states[-1].n, basis.states[-1].angular.l_r) == (20, 19)
    assert len(basis.states) == {"Rb": 388, "Na": 396, "H": 400}[species_name]

    state0 = basis.states[0]
    ov = basis.calc_reduced_overlap(state0)
    compare_ov = np.zeros(len(basis.states))
    compare_ov[0] = 1.0
    assert np.allclose(ov, compare_ov, atol=1e-3)

    me = basis.calc_reduced_matrix_element(state0, "electric_dipole", unit="e a0")
    assert np.shape(me) == (len(basis.states),)
    assert np.count_nonzero(me) > 0

    basis.filter_states("n", (1, 7))
    ov_matrix = basis.calc_reduced_overlaps(basis)
    assert np.allclose(ov_matrix, np.eye(len(basis.states)), atol=1e-3)

    me_matrix = basis.calc_reduced_matrix_elements(basis, "electric_dipole", unit="e a0")
    assert np.shape(me_matrix) == (len(basis.states), len(basis.states))
    assert np.count_nonzero(me_matrix) > 0


@pytest.mark.parametrize("species_name", ["Sr88", "Sr87", "Yb174", "Yb171"])
def test_alkaline_basis(species_name: str) -> None:
    """Test alkaline basis creation."""
    basis = BasisSQDT(species_name, n=(30, 35), coupling_scheme="LS")
    basis.sort_states("n", "l_r")
    assert (basis.states[0].n, basis.states[0].angular.l_r) == (30, 0)
    assert (basis.states[-1].n, basis.states[-1].angular.l_r) == (35, 34)
    assert len(basis.states) == {"Sr88": 768, "Sr87": 7188, "Yb174": 768, "Yb171": 1524}[species_name]

    # also test JJ and FJ coupling
    basis_jj = BasisSQDT(species_name, n=(30, 35), coupling_scheme="JJ")
    basis_fj = BasisSQDT(species_name, n=(30, 35), coupling_scheme="FJ")
    assert len(basis_jj.states) == len(basis.states)
    assert len(basis_fj.states) == len(basis.states)

    if species_name in ["Sr87", "Yb171"]:
        pytest.skip("Quantum defects for Sr87 and Yb171 not implemented yet.")

    state0 = basis.states[0]
    ov = basis.calc_reduced_overlap(state0)
    compare_ov = np.zeros(len(basis.states))
    compare_ov[0] = 1.0
    assert np.allclose(ov, compare_ov, atol=1e-3)

    me = basis.calc_reduced_matrix_element(state0, "electric_dipole", unit="e a0")
    assert np.shape(me) == (len(basis.states),)
    assert np.count_nonzero(me) > 0

    basis.filter_states("l_r", (0, 2))
    ov_matrix = basis.calc_reduced_overlaps(basis)
    assert np.allclose(ov_matrix, np.eye(len(basis.states)), atol=1e-2)

    me_matrix = basis.calc_reduced_matrix_elements(basis, "electric_dipole", unit="e a0")
    assert np.shape(me_matrix) == (len(basis.states), len(basis.states))
    assert np.count_nonzero(me_matrix) > 0
