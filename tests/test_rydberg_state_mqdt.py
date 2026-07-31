from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import BasisMQDT, RydbergStateSQDT
from rydstate.angular.utils import is_unknown
from rydstate.species import FModelSQDT

if TYPE_CHECKING:
    from rydstate import RydbergStateMQDT


@pytest.fixture(scope="module")
def basis() -> BasisMQDT:
    basis = BasisMQDT("Sr88", nu=(28.0, 30.0))
    basis.sort_states("nu")
    return basis


def _find_state(basis: BasisMQDT, *, l_r: int, f_tot: float, s_tot: float | None = None) -> RydbergStateMQDT:
    """Return the first MQDT state matching the given (expectation value) quantum numbers."""
    for state in basis.states:
        angular = state.angular
        if abs(angular.calc_exp_qn("l_r") - l_r) > 1e-6:
            continue
        if abs(angular.calc_exp_qn("f_tot") - f_tot) > 1e-6:
            continue
        if s_tot is not None and abs(angular.calc_exp_qn("s_tot") - s_tot) > 1e-2:
            continue
        return state
    raise AssertionError(f"No MQDT state found for l_r={l_r}, f_tot={f_tot}, s_tot={s_tot}")


def test_quantum_number_expectation_values(basis: BasisMQDT) -> None:
    """f_tot is a good quantum number; l_r/s_tot/j_tot have sensible expectation values."""
    for state in basis.states:
        angular = state.angular

        # f_tot is conserved within an MQDT model, so it is exact (zero spread).
        f_tot = angular.calc_exp_qn("f_tot")
        assert angular.calc_std_qn("f_tot") == 0
        # f_tot must be a (half-)integer >= 0.
        assert f_tot >= 0
        assert abs(2 * f_tot - round(2 * f_tot)) < 1e-12

        # l_r expectation value must lie within the range spanned by the channels.
        l_r_channels = [ket.angular.l_r for ket in state.rydberg_kets]
        l_r_exp = angular.calc_exp_qn("l_r")
        assert min(l_r_channels) <= l_r_exp <= max(l_r_channels)

        # s_tot expectation value is between 0 and 1 (singlet/triplet mixing).
        s_tot_exp = angular.calc_exp_qn("s_tot")
        assert 0 <= s_tot_exp <= 1


def test_single_channel_state_has_sharp_quantum_numbers(basis: BasisMQDT) -> None:
    """A single-channel MQDT state behaves like a pure SQDT-like state with sharp qn."""
    state = next(s for s in basis.states if len(s.rydberg_kets) == 1)
    angular = state.angular
    for qn in ("l_r", "j_r", "f_c", "f_tot", "s_tot", "j_tot"):
        assert angular.calc_std_qn(qn) < 1e-12


def test_self_overlap_is_one(basis: BasisMQDT) -> None:
    """Every MQDT state overlaps with itself to one."""
    for state in basis.states[:10]:
        assert np.isclose(state.calc_reduced_overlap(state), 1.0)


def test_overlap_matrix_is_approximately_identity(basis: BasisMQDT) -> None:
    """Distinct MQDT states from one basis are (approximately) mutually orthonormal.

    The diagonal is exactly one, while off-diagonal overlaps are only approximately
    zero: states sharing a channel but having different nu retain a small (~1e-2)
    radial overlap.
    """
    overlaps = basis.calc_reduced_overlaps(basis)
    assert overlaps.shape == (len(basis), len(basis))
    # Diagonal entries are exactly one.
    assert np.allclose(np.diag(overlaps), 1.0, atol=1e-12)
    # Off-diagonal entries are small.
    off_diagonal = overlaps - np.diag(np.diag(overlaps))
    assert np.abs(off_diagonal).max() < 1e-2


def test_overlap_with_sqdt_state(basis: BasisMQDT) -> None:
    """A single-channel triplet-S MQDT state overlaps with the matching SQDT state.

    For the (nearly) single-channel triplet-S series of Sr88, the MQDT quantum defect
    matches the SQDT one, so the SQDT state with the same nu has (almost) unit overlap.
    """
    mqdt_state = _find_state(basis, l_r=0, f_tot=1.0, s_tot=1.0)

    overlaps = {}
    for n in range(30, 35):
        sqdt_state = RydbergStateSQDT(mqdt_state.species, n=n, l_r=0, s_tot=1, l_tot=0, j_tot=1, f_tot=1)
        overlaps[n] = abs(mqdt_state.calc_reduced_overlap(sqdt_state))

    best_n = max(overlaps, key=overlaps.__getitem__)
    best_sqdt = RydbergStateSQDT(mqdt_state.species, n=best_n, l_r=0, s_tot=1, l_tot=0, j_tot=1, f_tot=1)

    # The best matching SQDT state has the same nu and (almost) unit overlap.
    assert np.isclose(best_sqdt.nu, mqdt_state.nu, atol=1e-2)
    assert overlaps[best_n] > 0.99
    # Non-matching nu give negligible overlap.
    for n, ov in overlaps.items():
        if n != best_n:
            assert ov < 1e-3


def test_dipole_matrix_element_between_mqdt_states(basis: BasisMQDT) -> None:
    """Electric-dipole matrix elements: S<->P allowed, S<->S forbidden."""
    s_state = _find_state(basis, l_r=0, f_tot=1.0, s_tot=1.0)
    p_state = _find_state(basis, l_r=1, f_tot=2.0, s_tot=1.0)

    # Allowed S <-> P transition: large reduced dipole matrix element (~ nu^2 a0).
    me_sp = s_state.calc_reduced_matrix_element(p_state, "electric_dipole", unit="e a0")
    assert abs(me_sp) > 1.0
    # Hermiticity (up to sign convention): |<s|d|p>| == |<p|d|s>|.
    me_ps = p_state.calc_reduced_matrix_element(s_state, "electric_dipole", unit="e a0")
    assert np.isclose(abs(me_sp), abs(me_ps))

    # Forbidden S <-> S transition (no parity change): vanishing dipole matrix element.
    other_s = _find_state(basis, l_r=0, f_tot=0.0)
    me_ss = s_state.calc_reduced_matrix_element(other_s, "electric_dipole", unit="e a0")
    assert abs(me_ss) < 1e-9


def test_matrix_element_between_mqdt_and_sqdt_state(basis: BasisMQDT) -> None:
    """A dipole matrix element can be computed between an MQDT and an SQDT state."""
    s_mqdt = _find_state(basis, l_r=0, f_tot=1.0, s_tot=1.0)
    p_sqdt = RydbergStateSQDT(s_mqdt.species, n=round(s_mqdt.nu) + 3, l_r=1, s_tot=1, l_tot=1, j_tot=2, f_tot=2)

    me = s_mqdt.calc_reduced_matrix_element(p_sqdt, "electric_dipole", unit="e a0")
    assert np.isfinite(me)
    assert abs(me) > 0.0


def test_n_of_sqdt_fallback_model_uses_channel_nui() -> None:
    """For SQDT fallback models n is given by the channel nui, not by nu."""
    basis = BasisMQDT("Yb171", nu=(78.0, 81.0), l_r=(5, 5), m=(0.5, 0.5))
    assert len(basis.states) > 0

    states_with_shifted_nu = 0
    for state in basis.states:
        assert isinstance(state.model, FModelSQDT)
        assert len(state.rydberg_kets) == 1

        nui = state.nui[0]
        assert abs(nui - round(nui)) < 1e-9
        assert state.n == round(nui)

        if state.n != round(state.nu):
            states_with_shifted_nu += 1

    assert states_with_shifted_nu > 0


@pytest.mark.parametrize(
    ("species", "m", "l_r", "model_name"),
    [
        ("Yb174", (0, 0), (0, 0), "S J=0, nu > 2"),
        ("Yb174", (0, 0), (2, 2), "D J=2, nu > 5"),
        ("Yb171", (0.5, 0.5), (1, 1), "P F=1/2, nu > 5.7"),
        ("Yb171", (0.5, 0.5), (1, 1), "P F=3/2, nu > 10"),
    ],
)
def test_n_increments_by_one_along_a_rydberg_series(
    species: str, m: tuple[float, float], l_r: tuple[int, int], model_name: str
) -> None:
    """Within one Rydberg series of an MQDT model, n increases by exactly one from state to state.

    A single MQDT model describes several interleaved Rydberg series (one per channel), so the states
    of a model have to be grouped by their dominant channel before comparing consecutive n.
    """
    basis = BasisMQDT(species, nu=(40.0, 46.0), l_r=l_r, m=m)
    states = [state for state in basis.states if state.model.name == model_name]
    assert len(states) > 0

    def dominant_channel(state: RydbergStateMQDT) -> int:
        """Return the index of the channel with the largest coefficient (ignoring unknown channels)."""
        known = [(abs(coeff), i) for i, (coeff, ket) in enumerate(state) if not is_unknown(ket.angular.l_r)]
        return max(known)[1]

    states.sort(key=lambda state: (dominant_channel(state), state.nu))
    for channel, series in itertools.groupby(states, key=dominant_channel):
        n_list = [state.n for state in series]
        assert len(n_list) > 1, f"series of channel {channel} is too short to test"
        assert all(n_next - n == 1 for n, n_next in itertools.pairwise(n_list)), (
            f"n does not increase by one along the series of channel {channel}: {n_list}"
        )
