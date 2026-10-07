from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import BasisMQDT
from rydstate.angular import AngularKetFJ
from rydstate.angular.utils import NotSet, is_unknown
from rydstate.basis.basis_mqdt import get_mqdt_states_from_model
from rydstate.species import TrivialModel, get_mqdt, get_potential_class

if TYPE_CHECKING:
    from rydstate.species import MQDTModel


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


def _channel_coefficients_mqdt_jl(model: MQDTModel, nu: float) -> np.ndarray:
    """Channel coefficients of the MQDT state at nu, calculated as in MQDT.jl (Peper et al.).

    MQDT.jl uses M = diag(sin(beta)) + K diag(cos(beta)) with beta_i = pi (nu_i - l_i) (l_i = 0 for channels with
    unknown l_r) and the coefficients nu_i^(3/2) x_i, where x is the null vector of M.
    """
    nuis = model.calc_channel_nuis(nu)
    l_r = np.array([0 if is_unknown(ket.l_r) else ket.l_r for ket in model.outer_channels])
    beta = np.pi * (nuis - l_r)
    m_matrix = np.diag(np.sin(beta)) + model.calc_k_matrix(nu) @ np.diag(np.cos(beta))
    _u, _s, vt = np.linalg.svd(m_matrix)
    coefficients = vt[-1] * nuis ** (3 / 2)
    coefficients /= np.linalg.norm(coefficients)
    return coefficients * np.sign(coefficients[np.argmax(np.abs(coefficients))])  # type: ignore [no-any-return]


@pytest.mark.parametrize(
    ("species", "tag", "model_name", "nu_range"),
    [
        ("Yb174", None, "S J=0, nu > 2", (5.0, 12.0)),
        ("Yb174", None, "D J=2, nu > 5", (6.0, 12.0)),
        ("Sr88", "vaillant2024", "S J=1, nu > 3.4", (3.4, 12.0)),
        ("Sr88", "vaillant2024", "D J=2, nu > 5.7", (5.7, 12.0)),
    ],
)
def test_mqdt_channel_coefficients_coulomb_phase_convention(
    species: str, tag: str | None, model_name: str, nu_range: tuple[float, float]
) -> None:
    """The channel coefficients follow the usual Coulomb phase convention beta_i = pi (nu_i - l_i).

    This matters for models whose channels have Rydberg electrons with different parity (e.g. 6sns and 6pnp),
    where it determines the relative sign of the channel coefficients and thereby the interference of the Rydberg
    and inner valence electron contributions to electric multipole matrix elements.
    The coefficients must agree with those of MQDT.jl and mqdtfit (Vaillant et al.), with which the models were fitted.
    """
    model = next(model for model in get_mqdt(species, tag).models if model.name == model_name)
    l_r_parities = {round(ket.l_r) % 2 for ket in model.outer_channels if not is_unknown(ket.l_r)}
    assert len(l_r_parities) == 2, "the test requires a model with channels of different l_r parity"

    # expansion of each outer channel into the FJ kets, in the same order as in the states
    n_fj = [len(list(ket.to_state("FJ"))) for ket in model.outer_channels]
    coefficients_fj = np.array([coeff for ket in model.outer_channels for coeff, _ in ket.to_state("FJ")])

    states = get_mqdt_states_from_model(model, nu_range, NotSet, get_potential_class(species))
    assert len(states) > 3
    for state in states:
        expected = np.repeat(_channel_coefficients_mqdt_jl(model, state.nu), n_fj) * coefficients_fj
        np.testing.assert_allclose(state.coefficients, expected, atol=1e-8, err_msg=f"{model.full_name} {state.nu=}")
