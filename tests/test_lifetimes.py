from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import BasisMQDT, RydbergStateSQDT, RydbergStateSQDTAlkali
from rydstate.angular import AngularKetLS
from rydstate.species import ElementProperties, get_all_subclasses, get_element_properties, get_mqdt
from rydstate.species.potential import get_potential_class

if TYPE_CHECKING:
    from rydstate import RydbergStateMQDT

ALL_AVAILABLE_SPECIES = sorted([cls.species for cls in get_all_subclasses(ElementProperties)])


# Reference values from NIST Atomic Spectra Database (ASD), Einstein A coefficients:
#   H 2p -> 1s: A = 6.2648e8 s^-1
#   H 3p -> all lower: A_tot = 1.8971e8 s^-1  (A(3p->1s)=1.6725e8, A(3p->2s)=2.245e7)
#   H 3d -> 2p: A = 6.4651e7 s^-1
# Source: https://physics.nist.gov/PhysRefData/ASD/lines_form.html
@pytest.mark.parametrize(
    ("n", "l", "j", "expected_gamma"),
    [
        (2, 1, 1.5, 6.2648e8),
        (3, 1, 1.5, 1.8971e8),
        (3, 2, 2.5, 6.4651e7),
    ],
)
def test_hydrogen_textbook_lifetimes(n: int, l: int, j: float, expected_gamma: float) -> None:
    """Test that calculated H lifetimes match NIST textbook values within 5%."""
    state = RydbergStateSQDTAlkali("H_textbook", n=n, l=l, j=j, m=0.5)
    tau = state.get_lifetime(unit="s")
    np.testing.assert_allclose(tau, 1 / expected_gamma, rtol=0.05)


def test_bbr_shortens_lifetime() -> None:
    """Test that black body radiation at 300 K shortens the lifetime relative to T=0."""
    state = RydbergStateSQDTAlkali("Rb", n=30, l=0, j=0.5, m=0.5)
    tau_0 = state.get_lifetime(unit="mus")
    tau_300 = state.get_lifetime(300, temperature_unit="K", unit="mus")
    assert tau_300 < tau_0


@pytest.mark.parametrize("species", ALL_AVAILABLE_SPECIES)
def test_lifetime_n_scaling(species: str) -> None:
    """Test that Rydberg state lifetimes scale as nu^3 (effective quantum number)."""
    if species != "Na":
        pytest.skip("Skip this test for most species for now, since this test is rather slow.")

    if species in ["Sr87", "Yb171", "Yb173"]:
        pytest.skip("No quantum defect data available")
    if species != "Yb174":
        pytest.skip("Quantum defects not correct for low n states")

    n1, n2 = 30, 60
    element_properties = get_element_properties(species)
    s_tot = (element_properties.number_valence_electrons / 2) % 1
    f = abs(s_tot - element_properties.i_c)
    angular = AngularKetLS(l_r=0, s_tot=s_tot, f_tot=f, m=f, species=species)

    state1 = RydbergStateSQDT(species, n=n1, angular=angular)
    state2 = RydbergStateSQDT(species, n=n2, angular=angular)
    tau1 = state1.get_lifetime(unit="mus")
    tau2 = state2.get_lifetime(unit="mus")
    nu1 = state1.nu
    nu2 = state2.nu
    expected_ratio = (nu2 / nu1) ** 3
    np.testing.assert_allclose(tau2 / tau1, expected_ratio, rtol=0.05)


def _get_mqdt_state(nu_range: tuple[float, float]) -> RydbergStateMQDT:
    """Return a single Sr88 S (l_r=0, f_tot=1) MQDT state in the given nu range."""
    basis = BasisMQDT("Sr88", nu=nu_range, l_r=(0, 0), f_tot=(1, 1), m=(0, 0))
    assert len(basis.states) >= 1
    return basis.states[0]


def test_mqdt_lifetime_is_finite_and_positive() -> None:
    """The spontaneous lifetime of an MQDT state can be computed and is finite and positive."""
    state = _get_mqdt_state((29.5, 29.8))
    tau = state.get_lifetime(unit="mus")
    assert np.isfinite(tau)
    assert tau > 0


def test_mqdt_bbr_shortens_lifetime() -> None:
    """Black body radiation at 300 K shortens the lifetime of an MQDT state relative to T=0."""
    state = _get_mqdt_state((29.5, 29.8))
    tau_0 = state.get_lifetime(unit="mus")
    tau_300 = state.get_lifetime(300, temperature_unit="K", unit="mus")
    assert tau_300 < tau_0


def test_mqdt_lifetime_nu_scaling() -> None:
    """MQDT Rydberg state lifetimes scale as nu^3 (effective quantum number)."""
    state1 = _get_mqdt_state((29.5, 29.8))
    state2 = _get_mqdt_state((49.5, 49.8))
    tau1 = state1.get_lifetime(unit="mus")
    tau2 = state2.get_lifetime(unit="mus")
    expected_ratio = (state2.nu / state1.nu) ** 3
    np.testing.assert_allclose(tau2 / tau1, expected_ratio, rtol=0.05)


def test_mqdt_transition_rate_basis_reuses_state_model_and_potential() -> None:
    """The internal basis used for transition rates reuses the state's own mqdt model and potential."""
    state = _get_mqdt_state((29.5, 29.8))

    # the state retains the model and potential it was built from ...
    assert state.mqdt is get_mqdt("Sr88")
    assert state.potential_class is get_potential_class("Sr88")

    # ... and these are forwarded into the basis built inside _get_transition_rates_au
    relevant_states, _ = state.get_spontaneous_transition_rates()
    assert len(relevant_states) > 0
    assert all(other.mqdt is state.mqdt for other in relevant_states)
    assert all(other.potential_class is state.potential_class for other in relevant_states)
