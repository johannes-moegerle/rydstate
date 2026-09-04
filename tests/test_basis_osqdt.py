from __future__ import annotations

import logging

import numpy as np
import pytest
from rydstate import BasisMQDT, BasisOSQDT, BasisSQDT
from rydstate.angular.utils import is_unknown


def test_osqdt_states_are_single_channel_with_definite_n() -> None:
    """Each OSQDT state consists of a single channel and has a definite principal quantum number n."""
    basis = BasisOSQDT("Yb174", nu=(50, 53), f_tot=(0, 0), l_r=(0, 0))
    assert len(basis) > 0
    for state in basis.states:
        assert len(state.rydberg_kets) == 1
        assert abs(state.norm - 1) < 1e-10
        assert isinstance(state.n, int)
        assert state.n > state.rydberg_kets[0].angular.l_r
        # nui = n - mu, with the quantum defect mu of the 6sns series of Yb being roughly 4.4
        assert 4 < state.n - state.nui < 5


def test_osqdt_covers_each_n_of_the_reference_channel_exactly_once() -> None:
    """For the channel at the reference threshold there must be exactly one state per n.

    The states of the series are spaced by one in nu, so the requested nu range must be covered by
    consecutive n (which are shifted with respect to nu by the quantum defect of the series).
    """
    nu_range = (50.0, 53.0)
    basis = BasisOSQDT("Yb174", nu=nu_range, f_tot=(0, 0), l_r=(0, 0)).filter_states("l_c", 0)
    ns = sorted(state.n for state in basis.states)
    assert len(ns) == 3
    assert ns == list(range(ns[0], ns[0] + len(ns)))
    assert all(nu_range[0] <= state.nu <= nu_range[1] for state in basis.states)


def test_osqdt_states_solve_the_decoupled_mqdt_condition() -> None:
    """The OSQDT states must solve the MQDT condition with the coupling between the channels switched off.

    Neglecting the off-diagonal elements of the K-matrix makes the M-matrix diagonal, so its determinant
    vanishes exactly when the diagonal element of the channel of the state vanishes.
    """
    basis = BasisOSQDT("Yb174", nu=(50, 53), f_tot=(0, 0), l_r=(0, 0))
    assert len(basis) > 0
    for state in basis.states:
        osqdt_models = [m for m in state.sqdt.osqdt_models if state.n in m.solutions]
        assert len(osqdt_models) == 1
        index = osqdt_models[0].index
        assert abs(osqdt_models[0].model.calc_scaled_m_matrix(state.nu)[index, index]) < 1e-9


def test_osqdt_l_r_filter() -> None:
    """Only channels with l_r inside the given range are included."""
    basis = BasisOSQDT("Yb174", nu=(50, 52), f_tot=(1, 1), l_r=(1, 1))
    assert len(basis) > 0
    for state in basis.states:
        assert state.rydberg_kets[0].angular.l_r == 1


@pytest.mark.parametrize("species", ["Yb171", "Yb174"])
def test_osqdt_contains_perturber_states_with_definite_n(species: str) -> None:
    """The perturber channels (unknown l_r) are included and get a definite n as well.

    Their core state is unknown, but their eigen quantum defects carry the integer part of the 6snl
    shell, so they are all labeled n=6, even though they lie in the middle of the Rydberg series
    of the other channels.
    """
    basis = BasisOSQDT(species, nu=(1, 60), f_tot=(0.5, 0.5) if species == "Yb171" else (1, 1))
    perturbers = basis.shallow_copy().filter_states_label("4f13 5d 6snl")
    assert len(perturbers) > 0
    for state in perturbers.states:
        assert len(state.rydberg_kets) == 1
        assert isinstance(state.n, int)
        assert state.n == 6
        assert state.nu > 1


def test_osqdt_overlaps_with_mqdt_are_normalized() -> None:
    """The MQDT states must be (almost) fully described by the OSQDT single channel states."""
    basis_mqdt = BasisMQDT("Yb174", nu=(50, 53), f_tot=(0, 0), l_r=(0, 0))
    # the perturber channels only support a single state each, which lies at a much smaller nu than
    # the MQDT states it perturbs, so the OSQDT basis must cover the whole nu range below them
    basis_osqdt = BasisOSQDT("Yb174", nu=(1, 60), f_tot=(0, 0))

    overlaps = basis_mqdt.calc_reduced_overlaps(basis_osqdt)
    np.testing.assert_allclose(np.sum(overlaps**2, axis=1), 1.0, atol=0.05)


@pytest.mark.parametrize(("l_r", "f_tot", "nu_range"), [(5, 5, (6, 10)), (5, 5, (1, 10)), (2, 2, (5, 8))])
def test_osqdt_n_coverage_matches_sqdt(l_r: int, f_tot: float, nu_range: tuple[float, float]) -> None:
    """For the channels built on the ground state core, OSQDT must cover the same n as BasisSQDT.

    Since the nu range translates into a different n range for each channel (n = nu + mu), the n range
    of BasisSQDT is taken from the OSQDT states, which must then contain every n of that range exactly once.
    This especially covers the states whose root lies exactly on the lower nu bound of the model
    (which is the case for the hydrogen-like channels with a vanishing quantum defect).
    """
    basis_osqdt = BasisOSQDT("Yb174", nu=nu_range, l_r=(l_r, l_r), f_tot=(f_tot, f_tot)).filter_states("l_c", 0)
    ns = sorted(state.n for state in basis_osqdt.states)
    assert len(ns) > 0
    basis_sqdt = BasisSQDT("Yb174", n=(ns[0], ns[-1]), l_r=(l_r, l_r), f_tot=(f_tot, f_tot))

    assert ns == sorted(state.n for state in basis_sqdt.states)


@pytest.mark.parametrize(("species", "f_tot"), [("Yb171", 0.5), ("Yb171", 1.5), ("Yb174", 2.0)])
def test_osqdt_covers_the_channels_of_an_excited_core(species: str, f_tot: float) -> None:
    """The channels built on an excited core must be taken from the MQDT models directly.

    Since the channels are constructed by looping over the FJ quantum numbers of the ground state core,
    the channels built on an excited core (which are partly given in the JJ or LS coupling scheme)
    cannot be constructed this way.
    """
    basis = BasisOSQDT(species, nu=(1, 60), f_tot=(f_tot, f_tot))
    channels = [state.rydberg_kets[0].angular for state in basis.states]

    assert all(channel.coupling_scheme == "FJ" for channel in channels)
    assert any(not is_unknown(channel.l_c) and channel.l_c > 0 for channel in channels)


@pytest.mark.parametrize(
    ("species", "l_r", "f_tot", "n_lowest"),
    [("Yb174", 0, 0, 6), ("Yb174", 1, 1, 6), ("Yb174", 2, 2, 5), ("Sr88", 1, 1, 5)],
)
def test_osqdt_lowest_state_of_a_series(species: str, l_r: int, f_tot: float, n_lowest: int) -> None:
    """The lowest state of a series must have the principal quantum number of its lowest allowed shell.

    For the series built on the ground state core this is n=6 for Yb and n=5 for Sr
    (and n=5 for the Yb 6snd series, since 5d is an additional allowed shell).
    This only works because the eigen quantum defects of the models carry their integer part.
    """
    basis = BasisOSQDT(species, nu=(1, 12), l_r=(l_r, l_r), f_tot=(f_tot, f_tot)).filter_states("l_c", 0)
    assert len(basis) > 0
    assert min(state.n for state in basis.states) == n_lowest


def test_osqdt_sanity_check_passes_for_a_complete_series(caplog: pytest.LogCaptureFixture) -> None:
    """A channel which is fully covered by its model must not be reported by the sanity check."""
    with caplog.at_level(logging.WARNING):
        basis = BasisOSQDT("Yb174", nu=(10, 20), l_r=(0, 0), f_tot=(0, 0))

    assert len(basis) > 0
    assert [record.message for record in caplog.records] == []


def test_osqdt_sanity_check_reports_a_gap(caplog: pytest.LogCaptureFixture) -> None:
    """The Sr88 5snp channels have no model between nu = 2.2 and nu = 5, which must be reported."""
    with caplog.at_level(logging.WARNING):
        BasisOSQDT("Sr88", nu=(1, 20), l_r=(1, 1), f_tot=(1, 1))

    assert all(
        any(f"No OSQDT state found for intermediate n={n}" in record.message for record in caplog.records)
        for n in [6, 7]
    )


def test_osqdt_sanity_check_reports_a_missing_low_nu_model(caplog: pytest.LogCaptureFixture) -> None:
    """The lowest Sr88 5snd model only starts at nu = 17, so its states start way above 5s4d."""
    with caplog.at_level(logging.WARNING):
        BasisOSQDT("Sr88", nu=(1, 40), l_r=(2, 2), f_tot=(1, 1))

    assert any("has no model for nui<17" in record.message for record in caplog.records), str(
        [record.message for record in caplog.records]
    )
