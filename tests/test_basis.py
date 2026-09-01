import numpy as np
import pytest
from rydstate import BasisSQDT


@pytest.mark.parametrize("species", ["Rb", "Na", "H"])
def test_alkali_basis(species: str) -> None:
    """Test alkali basis creation."""
    basis = BasisSQDT(species, n=(1, 20))
    basis.sort_states("n", "l_r")
    lowest_n_state = {"Rb": (4, 2), "Na": (3, 0), "H": (1, 0)}[species]
    assert (basis.states[0].n, basis.states[0].angular.l_r) == lowest_n_state
    assert (basis.states[-1].n, basis.states[-1].angular.l_r) == (20, 19)
    assert len(basis.states) == {"Rb": 388, "Na": 396, "H": 400}[species]

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


def test_sqdt_basis_m_range() -> None:
    basis = BasisSQDT("H", n=(5, 5), f_tot=(0.5, 0.5), l_r=(0, 0), m=(-0.5, 0.5))
    invalid_basis = BasisSQDT("H", n=(5, 5), f_tot=(0.5, 0.5), l_r=(0, 0), m=(0, 0))

    assert [state.angular.m for state in basis.states] == [-0.5, 0.5]
    assert len(invalid_basis.states) == 0


@pytest.mark.parametrize("species", ["Sr88", "Yb174"])
def test_alkaline_basis(species: str) -> None:
    """Test alkaline basis creation."""
    basis = BasisSQDT(species, n=(30, 35))
    basis.sort_states("n", "l_r")
    assert (basis.states[0].n, basis.states[0].angular.l_r) == (30, 0)
    assert (basis.states[-1].n, basis.states[-1].angular.l_r) == (35, 34)
    assert len(basis.states) == {"Sr88": 768, "Yb174": 768}[species]

    if species in ["Sr87", "Yb171"]:
        pytest.skip("Quantum defects for Sr87 and Yb171 not implemented yet.")

    state_ind = 2
    state = basis.states[state_ind]
    ov = basis.calc_reduced_overlap(state)
    compare_ov = np.zeros(len(basis.states))
    compare_ov[state_ind] = 1.0
    assert np.allclose(ov, compare_ov, atol=1e-3)

    me = basis.calc_reduced_matrix_element(state, "electric_dipole", unit="e a0")
    assert np.shape(me) == (len(basis.states),)
    assert np.count_nonzero(me) > 0

    basis.filter_states("l_r", (0, 2))
    ov_matrix = basis.calc_reduced_overlaps(basis)
    assert np.allclose(ov_matrix, np.eye(len(basis.states)), atol=1e-2)

    me_matrix = basis.calc_reduced_matrix_elements(basis, "electric_dipole", unit="e a0")
    assert np.shape(me_matrix) == (len(basis.states), len(basis.states))
    assert np.count_nonzero(me_matrix) > 0


def test_shallow_copy() -> None:
    """A shallow copy has an independent states list but shares the state objects."""
    basis = BasisSQDT("H", n=(1, 5))
    copied = basis.shallow_copy()
    assert copied.states is not basis.states
    assert all(a is b for a, b in zip(copied.states, basis.states, strict=True))

    n_states = len(basis)
    copied.filter_states("l_r", 0)
    assert len(copied) < n_states
    assert len(basis) == n_states


def test_filter_states_parity() -> None:
    """filter_states("parity", ...) splits the basis into even and odd states."""
    basis = BasisSQDT("H", n=(1, 5))
    even = basis.shallow_copy().filter_states("parity", 1)
    odd = basis.shallow_copy().filter_states("parity", -1)
    assert len(even) + len(odd) == len(basis)
    assert all(state.angular.l_r % 2 == 0 for state in even.states)
    assert all(state.angular.l_r % 2 == 1 for state in odd.states)
