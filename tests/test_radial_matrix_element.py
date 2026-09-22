import numpy as np
import pytest
from rydstate import RydbergStateSQDTAlkali
from rydstate.angular import AngularKetLS
from rydstate.radial import RadialKet
from rydstate.species import get_sqdt
from rydstate.species.potential import get_potential_class


@pytest.mark.parametrize(
    ("species", "n", "dn", "dl"),
    [
        ("Rb", 100, 3, 1),
        ("Rb", 60, 2, 0),
        ("Rb", 81, 2, 2),
        ("Rb", 130, 5, 1),
        ("Rb", 111, 5, 2),
        ("Cs", 60, 2, 0),
        ("K", 81, 2, 2),
    ],
)
def test_circular_matrix_element(species: str, n: int, dn: int, dl: int) -> None:
    """Test radial matrix elements of ((almost) circular states, i.e. with large l (l = n-1 for circular states).

     Circular matrix elements should be very close to the perfect hydrogen case, so we can check if the matrix elements
    are reasonable by comparing them to the hydrogen case.
    """
    l1 = n - 1  # circular state
    l2 = l1 + dl  # almost circular state

    matrix_element = {}
    for _species in [species, "H_textbook"]:
        state_i = RydbergStateSQDTAlkali(_species, n=n, l=l1, j=l1 + 0.5)
        state_f = RydbergStateSQDTAlkali(_species, n=n + dn, l=l2, j=l2 + 0.5)
        matrix_element[_species] = state_i.radial.calc_matrix_element(state_f.radial, 1, unit="bohr")

    assert np.isclose(matrix_element[species], matrix_element["H_textbook"], rtol=1e-4)


@pytest.mark.parametrize(
    ("species", "n", "l", "j_tot"),
    [
        # for hydrogen the expectation value of r is exact for all states
        ("H", 1, 0, 0.5),
        ("H", 2, 0, 0.5),
        ("H", 2, 1, 0.5),
        ("H", 2, 1, 1.5),
        ("H", 60, 30, 29.5),
        # for other species it is only approximate for circular states
        ("Rb", 100, 99, 99.5),
        ("Rb", 88, 87, 86.5),
    ],
)
def test_circular_expectation_value(species: str, n: int, l: int, j_tot: float) -> None:
    """For circular states, the expectation value of r should be the same as for the hydrogen atom.

    For hydrogen the expectation values of r and r^2 are given by

    .. math::
        <r>_{nl} = 1/2 (3 n^2 - l(l+1))
        <r^2>_{nl} = n^2/2 (5 n^2 - 3 l(l+1) + 1)
    """
    sqdt = get_sqdt(species)
    angular_ket = AngularKetLS(l_r=l, j_tot=j_tot, species=species)
    nu = sqdt.calc_nui(n, angular_ket)

    potential = get_potential_class(species)(l)
    state = RadialKet(nu, potential, n_expected=n)

    exp_value_numerov = {i: state.calc_matrix_element(state, i, unit=f"bohr^{i}" if i > 0 else "") for i in range(3)}
    exp_value_analytic = {
        0: 1,
        1: 0.5 * (3 * n**2 - l * (l + 1)),
        2: n**2 / 2 * (5 * n**2 - 3 * l * (l + 1) + 1),
    }

    for i in range(3):
        assert np.isclose(exp_value_numerov[i], exp_value_analytic[i], rtol=1e-2), (
            f"Expectation value of r^{i} is not correct."
        )


@pytest.mark.parametrize(
    ("n", "l", "j_tot"),
    [
        (2, 1, 1.5),
        (10, 3, 3.5),
        (60, 30, 29.5),
    ],
)
def test_expectation_value_of_powers_of_r(n: int, l: int, j_tot: float) -> None:
    r"""For hydrogen the expectation values of r^k are known analytically, for both positive and negative k.

    .. math::
        <r^{-3}>_{nl} = 1 / (n^3 l (l + 1/2) (l + 1))
        <r^{-2}>_{nl} = 1 / (n^3 (l + 1/2))
        <r^{-1}>_{nl} = 1 / n^2
        <r^{1}>_{nl} = 1/2 (3 n^2 - l(l+1))
        <r^{2}>_{nl} = n^2/2 (5 n^2 + 1 - 3 l(l+1))
        <r^{3}>_{nl} = n^2/8 (35 n^4 - 5 n^2 (6 l(l+1) - 5) + 3 (l+2)(l+1)l(l-1))
    """
    sqdt = get_sqdt("H")
    angular_ket = AngularKetLS(l_r=l, j_tot=j_tot, species="H")
    nu = sqdt.calc_nui(n, angular_ket)

    potential = get_potential_class("H")(l)
    state = RadialKet(nu, potential, n_expected=n)

    # note: pint cannot parse "bohr^0", so the dimensionless k = 0 case uses an empty unit
    exp_value_numerov = {
        k: state.calc_matrix_element(state, k, unit=f"bohr^{k}" if k != 0 else "") for k in range(-3, 4)
    }
    exp_value_analytic = {
        -3: 1 / (n**3 * l * (l + 0.5) * (l + 1)),
        -2: 1 / (n**3 * (l + 0.5)),
        -1: 1 / n**2,
        0: 1,
        1: 0.5 * (3 * n**2 - l * (l + 1)),
        2: n**2 / 2 * (5 * n**2 + 1 - 3 * l * (l + 1)),
        3: n**2 / 8 * (35 * n**4 - 5 * n**2 * (6 * l * (l + 1) - 5) + 3 * (l + 2) * (l + 1) * l * (l - 1)),
    }

    for k in range(-3, 4):
        assert np.isclose(exp_value_numerov[k], exp_value_analytic[k], rtol=1e-2), (
            f"Expectation value of r^{k} is not correct."
        )
