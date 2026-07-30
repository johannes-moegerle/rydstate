import logging

import pytest
from rydstate import BasisMQDT
from rydstate.angular import AngularKetFJ
from rydstate.species import FModelSQDT, get_mqdt


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
        assert abs(state.angular.calc_exp_qn("f_tot") - 0.0) < 1e-10


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
        include_sqdt_fallback_models=True,
    )

    expected_channels = {
        AngularKetFJ(l_r=5, j_r=4.5, f_c=4.0, f_tot=0.5, species="Sr87"),
        AngularKetFJ(l_r=5, j_r=4.5, f_c=5.0, f_tot=0.5, species="Sr87"),
        AngularKetFJ(l_r=5, j_r=5.5, f_c=5.0, f_tot=0.5, species="Sr87"),
    }

    assert all(isinstance(model, FModelSQDT) for model in basis.models)
    assert len(basis.models) == len(expected_channels)
    assert {model.outer_channels[0] for model in basis.models} == expected_channels


def test_mqdt_basis_without_sqdt_fallback_models_includes_all_available_mqdt_models() -> None:
    basis = BasisMQDT("Sr88", nu=(25, 30), include_sqdt_fallback_models=False)
    expected_models = get_mqdt("Sr88").models

    assert not any(isinstance(model, FModelSQDT) for model in basis.models)
    assert len(basis.models) == len(expected_models)
    assert {model.full_name for model in basis.models} == {model.full_name for model in expected_models}


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
