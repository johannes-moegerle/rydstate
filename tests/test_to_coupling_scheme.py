from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from rydstate import BasisMQDT, RydbergStateSQDT
from rydstate.angular import AngularKetLS

if TYPE_CHECKING:
    from rydstate.angular.utils import CouplingScheme


@pytest.fixture(scope="module")
def ls_state() -> RydbergStateSQDT[Any]:
    """Return a triplet-P (3P1) SQDT state of Sr88, given in the LS coupling scheme."""
    angular = AngularKetLS(l_r=1, s_tot=1, l_tot=1, j_tot=1, f_tot=1, species="Sr88")
    return RydbergStateSQDT("Sr88", n=60, angular=angular)


@pytest.fixture(scope="module")
def mqdt_basis() -> BasisMQDT:
    return BasisMQDT("Sr88", nu=(28.0, 30.0))


# --------------------------------------------------------------------------------------
# basic invariants of the conversion
# --------------------------------------------------------------------------------------


def test_converting_to_same_scheme_is_identity(ls_state: RydbergStateSQDT[Any]) -> None:
    converted = ls_state.to_coupling_scheme("LS")

    assert len(converted.rydberg_kets) == 1
    assert np.allclose(converted.coefficients, [1.0])
    assert np.isclose(ls_state.calc_reduced_overlap(converted), 1.0)


def test_conversion_preserves_norm_energy_and_nu(
    ls_state: RydbergStateSQDT[Any], coupling_scheme: CouplingScheme
) -> None:
    converted = ls_state.to_coupling_scheme(coupling_scheme)

    assert np.isclose(converted.norm, 1.0)
    assert converted.species == ls_state.species
    assert np.isclose(converted.nu, ls_state.nu)
    assert np.isclose(converted.get_energy("a.u."), ls_state.get_energy("a.u."))


def test_converted_kets_live_in_the_target_scheme(
    ls_state: RydbergStateSQDT[Any], coupling_scheme: CouplingScheme
) -> None:
    converted = ls_state.to_coupling_scheme(coupling_scheme)
    assert all(ket.angular.coupling_scheme == coupling_scheme for ket in converted.rydberg_kets)


def test_conversion_preserves_physical_state(ls_state: RydbergStateSQDT[Any], coupling_scheme: CouplingScheme) -> None:
    """A coupling-scheme change is a unitary rotation of the angular basis: the state is unchanged."""
    converted = ls_state.to_coupling_scheme(coupling_scheme)
    assert np.isclose(ls_state.calc_reduced_overlap(converted), 1.0)


def test_good_quantum_number_expectation_value_is_preserved(
    ls_state: RydbergStateSQDT[Any], coupling_scheme: CouplingScheme
) -> None:
    # f_tot is a good quantum number, so it must survive the basis rotation
    converted = ls_state.to_coupling_scheme(coupling_scheme)
    assert np.isclose(converted.calc_exp_qn("f_tot"), ls_state.calc_exp_qn("f_tot"))


# --------------------------------------------------------------------------------------
# the concrete LS <-> JJ expansion
# --------------------------------------------------------------------------------------


def test_ls_to_jj_expands_with_clebsch_gordan_coefficients(ls_state: RydbergStateSQDT[Any]) -> None:
    """3P1 in LS is a superposition of two JJ channels (j_r=1/2 and j_r=3/2)."""
    jj = ls_state.to_coupling_scheme("JJ")

    assert len(jj.rydberg_kets) == 2
    # coefficients are sqrt(2/3) and sqrt(1/3) (up to ordering)
    assert np.allclose(sorted(jj.coefficients), sorted([np.sqrt(1 / 3), np.sqrt(2 / 3)]))
    assert np.isclose(np.sum(jj.coefficients**2), 1.0)


def test_roundtrip_recovers_original_state(ls_state: RydbergStateSQDT[Any]) -> None:
    roundtrip = ls_state.to_coupling_scheme("JJ").to_coupling_scheme("LS")

    # the physical state is recovered exactly (up to zero-weight channels that the
    # back-conversion may spuriously generate, e.g. the singlet 1P1 partner of 3P1)
    assert np.isclose(ls_state.calc_reduced_overlap(roundtrip), 1.0)
    assert np.isclose(np.max(roundtrip.coefficients), 1.0)
    assert np.isclose(np.sum(roundtrip.coefficients**2), 1.0)


def test_cancelling_channels_are_dropped(ls_state: RydbergStateSQDT[Any]) -> None:
    """The near-cancellation guard (radial norm < 1e-12) must drop spuriously generated channels.

    Converting 3P1 (LS) to JJ and back re-expands each JJ channel into LS components; the singlet
    1P1 partner of 3P1 cancels to ~0 across the two JJ channels. Without the guard the conversion
    would emit a bogus extra ket with a numerically-zero radial wavefunction, so asserting the exact
    ket count pins the guard down (this test fails if the ``norm < 1e-12`` skip is removed).
    """
    roundtrip = ls_state.to_coupling_scheme("JJ").to_coupling_scheme("LS")

    assert len(roundtrip.rydberg_kets) == 1


def test_calc_exp_qn_uses_conversion_for_foreign_quantum_number(ls_state: RydbergStateSQDT[Any]) -> None:
    """`j_r` is not defined in the LS scheme, so calc_exp_qn must convert to JJ internally."""
    assert "j_r" not in ls_state.rydberg_kets[0].angular.quantum_number_names

    exp_via_calc = ls_state.calc_exp_qn("j_r")
    exp_via_manual_conversion = ls_state.to_coupling_scheme("JJ").calc_exp_qn("j_r")
    assert np.isclose(exp_via_calc, exp_via_manual_conversion)


# --------------------------------------------------------------------------------------
# multi-channel (MQDT) states: exercises the radial-combination (+=) branch
# --------------------------------------------------------------------------------------


def test_mqdt_multichannel_conversion_preserves_state(mqdt_basis: BasisMQDT, coupling_scheme: CouplingScheme) -> None:
    multichannel = [state for state in mqdt_basis.states if len(state.rydberg_kets) > 1]
    assert multichannel, "expected some multi-channel MQDT states in the basis"

    for state in multichannel:
        converted = state.to_coupling_scheme(coupling_scheme)
        assert np.isclose(converted.norm, 1.0)
        assert np.isclose(state.calc_reduced_overlap(converted), 1.0)
        assert all(ket.angular.coupling_scheme == coupling_scheme for ket in converted.rydberg_kets)


def test_mqdt_conversion_combines_shared_channels(mqdt_basis: BasisMQDT) -> None:
    """When independent channels map onto a shared angular ket, their radial wavefunctions are summed.

    Such a conversion produces fewer kets than the naive sum of the per-channel expansions, which
    exercises the radial-wavefunction summation (``+=``) branch of ``to_coupling_scheme``. This
    intrinsically scans all schemes at once, so it is not parametrized by the ``coupling_scheme`` fixture.
    """
    multichannel = [state for state in mqdt_basis.states if len(state.rydberg_kets) > 1]
    assert multichannel, "expected some multi-channel MQDT states in the basis"

    for state in multichannel:
        for scheme in ("LS", "JJ", "FJ"):
            converted = state.to_coupling_scheme(scheme)
            naive_number_of_kets = sum(len(ket.angular.to_state(scheme).kets) for ket in state.rydberg_kets)
            if len(converted.rydberg_kets) < naive_number_of_kets:
                return  # found a conversion that merged channels -> the += branch ran

    pytest.fail("expected at least one conversion to merge channels onto a shared angular ket")
