from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import BasisMQDT, RydbergStateSQDTAlkali
from rydstate.angular import AngularKetFJ
from rydstate.angular.utils import is_unknown
from rydstate.qdt import TrivialModel
from rydstate.radial import RadialKet

if TYPE_CHECKING:
    from rydstate.qdt import ModelMQDT


def test_mqdt_basis_creation() -> None:
    """Smoke-test that the basis builds and respects the nu range."""
    basis = BasisMQDT("Sr88", nu=(25, 30))
    assert len(basis) > 0
    assert all(25 <= s.nu <= 30 for s in basis.states)


def test_mqdt_basis_coefficients_normalized() -> None:
    """Every state must have unit-norm coefficients."""
    basis = BasisMQDT("Sr88", nu=(25, 30))
    for state in basis.states:
        assert abs(state.norm - 1) < 1e-6


def test_mqdt_basis_f_tot_filter() -> None:
    """When f_tot is passed, only states with that total angular momentum are returned."""
    basis = BasisMQDT("Sr88", nu=(25, 30), f_tot=(0, 0))
    assert len(basis) > 0
    for state in basis.states:
        assert abs(state.calc_exp_qn("f_tot") - 0.0) < 1e-10


def test_mqdt_basis_m_range() -> None:
    basis = BasisMQDT("Sr87", nu=(25, 26), f_tot=(3.5, 3.5), l_r=(0, 0), m=(-0.5, 0.5))

    assert len(basis.states) == 2
    assert [state.rydberg_kets[0].angular.m for state in basis.states] == [-0.5, 0.5]
    assert all(
        all(ket.angular.m == state.rydberg_kets[0].angular.m for ket in state.rydberg_kets) for state in basis.states
    )


def test_mqdt_basis_includes_all_available_sqdt_fallback_models() -> None:
    basis = BasisMQDT(
        "Sr87",
        nu=(24.9, 25.1),
        l_r=(5, 5),
        f_tot=(0.5, 0.5),
    )

    expected_channels = {
        AngularKetFJ(l_r=5, j_r=4.5, f_c=4.0, f_tot=0.5, species="Sr87"),
        AngularKetFJ(l_r=5, j_r=4.5, f_c=5.0, f_tot=0.5, species="Sr87"),
        AngularKetFJ(l_r=5, j_r=5.5, f_c=5.0, f_tot=0.5, species="Sr87"),
    }

    assert all(isinstance(model, TrivialModel) for model in basis.models)
    assert len(basis.models) == len(expected_channels)
    assert {model.outer_channels[0] for model in basis.models} == expected_channels


def test_mqdt_basis_includes_states_at_the_boundaries_of_the_nu_range() -> None:
    """States whose nu lies exactly on a boundary of the requested nu range are included.

    The quantum defects of the SQDT fallback models vanish, so their states lie at integer nu,
    i.e. exactly on the boundaries here. This only works if calc_channel_nuis resolves nu well
    enough for find_roots to locate these roots inside of the requested range.
    """
    basis = BasisMQDT("Yb171", nu=(30, 33), l_r=(5, 5))
    assert any(isinstance(model, TrivialModel) for model in basis.models)

    nus = [state.nu for state in basis.states]
    assert any(abs(nu - 30) < 1e-12 for nu in nus), f"no state at the lower boundary nu=30: {sorted(nus)[:5]}"
    assert any(abs(nu - 33) < 1e-12 for nu in nus), f"no state at the upper boundary nu=33: {sorted(nus)[-5:]}"


def test_mqdt_basis_sort_and_filter() -> None:
    """sort_states and filter_states work on MQDT states."""
    basis = BasisMQDT("Sr88", nu=(25, 30))
    basis.sort_states("nu")
    nus = [s.nu for s in basis.states]
    assert nus == sorted(nus)

    n_before = len(basis)
    basis.filter_states("nu", (27.0, 28.0))
    assert len(basis) < n_before
    assert all(27.0 - 1e-10 <= s.nu <= 28.0 + 1e-10 for s in basis.states)


def test_mqdt_basis_filter_by_label() -> None:
    """filter_states_label keeps only states with a matching channel label."""
    basis = BasisMQDT("Yb174", nu=(25, 26), f_tot=(0, 1))
    n_states = len(basis)
    assert n_states > 0

    labeled = basis.shallow_copy().filter_states_label("4f13 5d")
    assert 0 < len(labeled) < n_states
    for state in labeled.states:
        assert any(ket.angular.label is not None and "4f13 5d" in ket.angular.label for ket in state.rydberg_kets)


@pytest.mark.parametrize("nu", [86.97, 90.97, 92.97, 93.97, 95.97, 96.97, 97.97])
def test_difficult_yb174_g4_highn_states(nu: float, caplog: pytest.LogCaptureFixture) -> None:
    """Test that the some of the difficult states of the Yb174_G4_HighN model can be found in the basis.

    In former versions, calc_nullvector would fail to find the nullspace of the M-matrix for these states.
    """
    with caplog.at_level(logging.WARNING):
        basis = BasisMQDT("Yb174", nu=(nu - 0.2, nu + 0.2), l_r=(4, 4), f_tot=(4.0, 4.0))
        assert any(abs(s.nu - nu) < 1e-2 for s in basis.states)

    errors = [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert len(errors) == 0, "Unexpected errors were logged: " + "; ".join(record.getMessage() for record in errors)
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(warnings) == 0, "Unexpected warnings were logged: " + "; ".join(
        record.getMessage() for record in warnings
    )


@pytest.mark.parametrize(("n", "l"), [(1, 0), (2, 1), (3, 0), (3, 1), (3, 2), (6, 2), (7, 3), (10, 4)])
def test_coulomb_sign_near_origin_vs_outer_bound(n: int, l: int) -> None:
    """A hydrogen wavefunction positive at the outer bound has the sign (-1)^(n - l - 1) near the origin.

    The regular Coulomb function (positive near the origin) has the decaying part -cos(beta) u_- at large r with
    beta = pi (nu - l), i.e. the sign -cos(pi (n - l)) = (-1)^(n - l - 1) for nu = n.
    This checks the -l in beta, see test_mqdt_channel_coefficients_match_coulomb_asymptotics.
    """
    state = RydbergStateSQDTAlkali("H_textbook", n=n, l=l, j=l + 0.5)
    radial = RadialKet(state.nu, state.radial.potential, sign_convention="positive_at_outer_bound")
    assert radial.nodes == n - l - 1
    u_inner = radial.u_list[np.flatnonzero(np.abs(radial.u_list) > 1e-6 * np.max(np.abs(radial.u_list)))[0]]
    assert np.sign(u_inner) == (-1) ** (n - l - 1)


def _calc_expected_channel_coefficients(model: ModelMQDT, nu: float) -> np.ndarray:
    r"""Calculate the expected channel coefficients of an MQDT state from the Coulomb asymptotics.

    In the outer channel frame the state is psi_i = f_i b_i - g_i (K b)_i, where f_i (regular, positive near the
    origin) and g_i are the Coulomb functions in which K = tan(pi mu) is defined. For r -> infinity

        f -> sin(beta) u_+ - cos(beta) u_-,    g -> -cos(beta) u_+ - sin(beta) u_-,    beta = pi (nu - l)

    with the growing / decaying functions u_+ / u_- both positive and beta the phase f accumulates up to the outer
    turning point (the centrifugal barrier removes l half-periods, see test_coulomb_sign_near_origin_vs_outer_bound).
    Hence

        growing coefficient:   sin(beta_i) b_i + cos(beta_i) (K b)_i = 0   (bound state condition),
        decaying coefficient:  -cos(beta_i) b_i + sin(beta_i) (K b)_i,

    and the coefficient of the normalized radial ket (positive at the outer bound) is the decaying coefficient
    times nu_i^(3/2). Channels with unknown l_r get l = 0 (same convention as BasisMQDT).
    """
    nuis = model.calc_channel_nuis(nu)
    kmat = model.calc_k_matrix(nu)
    # the scaled M-matrix has the same nullspace as M = tan(pi nu) + K, but is better conditioned
    b = np.linalg.svd(model.calc_scaled_m_matrix(nu))[2][-1]
    ls = np.array([0 if is_unknown(ket.l_r) else int(ket.l_r) for ket in model.outer_channels])
    beta = np.pi * (nuis - ls)

    growing = np.sin(beta) * b + np.cos(beta) * (kmat @ b)
    decaying = -np.cos(beta) * b + np.sin(beta) * (kmat @ b)
    np.testing.assert_allclose(growing / np.linalg.norm(decaying), 0, atol=1e-6)

    expected: np.ndarray = nuis ** (3 / 2) * decaying
    expected /= np.linalg.norm(expected)
    return expected


@pytest.mark.parametrize(
    ("species", "f_tot", "model_name", "nu_range"),
    [
        ("Sr88_vaillant2024", 1, "S J=1, nu > 3.4", (4, 15)),
        ("Sr88_vaillant2024", 2, "D J=2, nu > 5.7", (6, 15)),
        ("Yb171", 0.5, "S F=1/2, 2 < nu < 26", (8, 20)),
        ("Yb171", 1.5, "D F=3/2, 2 < nu < 30", (20, 29)),
        ("Yb174", 0, "S J=0, nu > 2", (8, 20)),
    ],
)
def test_mqdt_channel_coefficients_match_coulomb_asymptotics(
    species: str, f_tot: float, model_name: str, nu_range: tuple[float, float]
) -> None:
    """The channel coefficients must agree with the decaying part of the Coulomb asymptotics, incl. its (-1)^l_r.

    The tested models couple channels whose l_r differ in parity, so the (-1)^l_r phase between the
    Coulomb functions (positive near the origin) and the radial kets (positive at the outer bound) matters.
    """
    basis = BasisMQDT(species, nu=nu_range, f_tot=(f_tot, f_tot))
    states = [state for state in basis.states if state.model.name == model_name]
    assert len(states) > 3

    for state in states:
        model = state.model
        # project the state onto the (orthonormal) outer channels to collapse the FJ kets back to the outer channels
        channel_coefficients = np.array(
            [
                sum(coeff * channel.calc_reduced_overlap(ket.angular) for coeff, ket in state)
                for channel in model.outer_channels
            ]
        )

        expected = _calc_expected_channel_coefficients(model, state.nu)
        # the global sign of a state is arbitrary, so we fix it via the largest component,
        # all relative signs between the channels must then agree
        i_max = np.argmax(np.abs(expected))
        global_sign = np.sign(channel_coefficients[i_max] * expected[i_max])
        np.testing.assert_allclose(channel_coefficients, global_sign * expected, atol=1e-6, err_msg=f"{state}")
