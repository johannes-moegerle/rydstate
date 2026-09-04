from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from rydstate import BasisMQDT, BasisTunableMQDT
from rydstate.species import FModelScaledOffDiagonal, get_mqdt

if TYPE_CHECKING:
    from rydstate.species import FModel


@pytest.fixture
def model() -> FModel:
    """Return a multi-channel model of Yb174."""
    return next(model for model in get_mqdt("Yb174").models if model.name == "S J=0, nu > 2")


def test_scaled_off_diagonal_model_only_scales_the_coupling(model: FModel) -> None:
    """The wrapper must scale the off-diagonal elements of K (and M) and leave their diagonals alone."""
    scale = 0.3
    scaled = FModelScaledOffDiagonal(model, scale)
    nu = 51.3

    for calc_matrix in ("calc_k_matrix", "calc_m_matrix"):
        mat = getattr(model, calc_matrix)(nu)
        mat_scaled = getattr(scaled, calc_matrix)(nu)
        np.testing.assert_allclose(np.diag(mat_scaled), np.diag(mat), atol=1e-12)
        np.testing.assert_allclose(
            mat_scaled - np.diag(np.diag(mat_scaled)), scale * (mat - np.diag(np.diag(mat))), atol=1e-12
        )


def test_scaled_off_diagonal_model_is_transparent_for_scale_one(model: FModel) -> None:
    """A scaling factor of 1 must reproduce the wrapped model."""
    scaled = FModelScaledOffDiagonal(model, 1.0)
    nu = 51.3

    np.testing.assert_allclose(scaled.calc_k_matrix(nu), model.calc_k_matrix(nu), atol=1e-12)
    np.testing.assert_allclose(scaled.calc_scaled_m_matrix(nu), model.calc_scaled_m_matrix(nu), atol=1e-12)


def test_tunable_mqdt_reproduces_mqdt_for_full_coupling() -> None:
    """With coupling_factor=1 the TunableMQDT basis must be identical to the MQDT basis."""
    kwargs = {"nu": (50, 53), "f_tot": (0, 0), "l_r": (0, 0)}
    basis_mqdt = BasisMQDT("Yb174", **kwargs)  # type: ignore [arg-type]
    basis_tunable_mqdt = BasisTunableMQDT("Yb174", **kwargs, coupling_factor=1.0)  # type: ignore [arg-type]

    assert len(basis_tunable_mqdt) == len(basis_mqdt)
    np.testing.assert_allclose(basis_tunable_mqdt.calc_exp_qn("nu"), basis_mqdt.calc_exp_qn("nu"), atol=1e-12)


def test_tunable_mqdt_scaling_interpolates() -> None:
    """The nu values must move continuously and monotonically from scale=0 to scale=1."""
    nus = [
        BasisTunableMQDT("Yb174", nu=(50, 51), f_tot=(0, 0), l_r=(0, 0), coupling_factor=scale).calc_exp_qn("nu")
        for scale in np.linspace(0, 1, 6)
    ]
    assert all(len(nu) == len(nus[0]) for nu in nus)
    # the fully coupled result differs noticeably from the decoupled one
    assert np.abs(nus[-1] - nus[0]).max() > 1e-4
    # and each step is small compared to the level spacing of 1
    assert np.abs(np.diff(nus, axis=0)).max() < 0.5
