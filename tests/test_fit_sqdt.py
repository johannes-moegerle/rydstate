from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from rydstate.fit import Measurement, SQDTLevel, fit_sqdt_models
from rydstate.fit.fit_sqdt import _calc_covariance
from rydstate.qdt import ModelSQDT, ModelSQDTFit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from rydstate.fit import FitResult

SIGMA_AU = 1e-12  # about 7 kHz

TRUE_NP = ModelSQDT("Cs", "np_3/2", [3.5590676, 0.37469])
TRUE_NS = ModelSQDT("Cs", "ns_1/2", [4.0493532, 0.2391])


def make_level_measurements(model: ModelSQDT, ns: range, dataset: str = "", shift_au: float = 0) -> list[Measurement]:
    return [
        Measurement(SQDTLevel(model.name, n), model.calc_energy_au(n=n) + shift_au, SIGMA_AU, dataset=dataset)
        for n in ns
    ]


def assert_fitted(result: FitResult, true_model: ModelSQDT) -> None:
    """Assert that the fitted quantum defects agree with the true ones far within their uncertainties."""
    deviations = np.abs(np.array(result.models[0].quantum_defects) - np.array(true_model.quantum_defects))
    sigmas = np.sqrt(np.diag(result.get_quantum_defect_covariance(true_model.name)))
    assert np.all(deviations < 1e-3 * sigmas)


def test_fit_recovers_quantum_defects() -> None:
    measurements = make_level_measurements(TRUE_NP, range(30, 80, 3))
    start = ModelSQDTFit("Cs", "np_3/2", [3.55, 0.3])
    result = fit_sqdt_models([start], measurements)

    assert result.success
    assert_fitted(result, TRUE_NP)
    assert result.chi2 == pytest.approx(0, abs=1e-6)
    assert result.dof == len(measurements) - 2
    assert start.quantum_defects == (3.55, 0.3)  # the start model is not changed
    assert result.models[0].reference is None  # the measurements have no reference
    assert len(result.parameter_names) == 2
    assert result.covariance.shape == (2, 2)
    # the start model is not changed
    assert start.quantum_defects == (3.55, 0.3)


def test_fit_intervals_with_fixed_model() -> None:
    """Intervals to levels of a fixed model determine the quantum defects of the fitted model."""
    measurements = [
        Measurement(
            SQDTLevel("ns_1/2", n + 4),
            TRUE_NS.calc_energy_au(n=n + 4) - TRUE_NP.calc_energy_au(n=n),
            SIGMA_AU,
            lower=SQDTLevel("np_3/2", n),
        )
        for n in range(40, 90, 5)
    ]
    result = fit_sqdt_models([ModelSQDTFit("Cs", "ns_1/2", [4.0, 0.2])], measurements, fixed_models=[TRUE_NP])

    assert len(result.models) == 1
    assert type(result.models[0]) is ModelSQDT
    assert_fitted(result, TRUE_NS)
    assert result.models[0].fit_info is not None
    assert result.models[0].fit_info["n range of the data"] == [44, 89]  # only the levels of the fitted model


def test_fit_free_offset() -> None:
    """A free offset of a dataset absorbs a common shift of its energies, but not of intervals."""
    shift_au = 3e-10
    measurements = make_level_measurements(TRUE_NP, range(30, 80, 3), dataset="shifted", shift_au=shift_au)
    result = fit_sqdt_models([ModelSQDTFit("Cs", "np_3/2", [3.55, 0.3])], measurements, free_offsets=["shifted"])

    assert abs(result.offsets_au["shifted"] + shift_au) < 1e-3 * result.offset_uncertainties_au["shifted"]
    assert result.offset_uncertainties_au["shifted"] == pytest.approx(np.sqrt(result.covariance[-1, -1]))
    assert_fitted(result, TRUE_NP)
    assert result.parameter_names[-1] == "offset shifted"


def test_fitted_models_have_fit_info() -> None:
    """The fitted models are ModelSQDT with where they came from, also in their json."""
    levels = [
        dataclasses.replace(m, reference="Levels 2026")
        for m in make_level_measurements(TRUE_NP, range(30, 80, 3), dataset="levels")
    ]
    intervals = [
        Measurement(
            SQDTLevel("ns_1/2", n + 4),
            TRUE_NS.calc_energy_au(n=n + 4) - TRUE_NP.calc_energy_au(n=n),
            SIGMA_AU,
            lower=SQDTLevel("np_3/2", n),
            dataset="intervals",
            reference="Intervals 2026",
        )
        for n in range(40, 90, 5)
    ]
    start = [ModelSQDTFit("Cs", "np_3/2", [3.55, 0.3]), ModelSQDTFit("Cs", "ns_1/2", [4.0, 0.2])]
    result = fit_sqdt_models(start, levels + intervals, free_offsets=["levels"])
    fitted_np, fitted_ns = result.models

    assert type(fitted_np) is ModelSQDT
    assert np.array_equal(result.get_quantum_defect_covariance("np_3/2"), result.covariance[:2, :2])
    assert np.array_equal(result.get_quantum_defect_covariance("ns_1/2"), result.covariance[2:4, 2:4])
    with pytest.raises(ValueError, match="0 fitted models"):
        result.get_quantum_defect_covariance("nd_5/2")
    assert fitted_np.reference == ("Levels 2026", "Intervals 2026")
    assert fitted_ns.reference == ("Intervals 2026",)
    assert fitted_np.fit_info is not None
    assert fitted_np.fit_info["n range of the data"] == [30, 85]
    assert set(fitted_np.fit_info) == {"method", "rydstate version", "n range of the data"}
    assert fitted_ns.fit_info is not None
    assert fitted_ns.fit_info["n range of the data"] == [44, 89]

    loaded = ModelSQDT.from_dict(json.loads(json.dumps(fitted_ns.to_dict())))
    assert loaded.quantum_defects == fitted_ns.quantum_defects
    assert loaded.fit_info == fitted_ns.fit_info


def test_fit_errors() -> None:
    measurements = make_level_measurements(TRUE_NP, range(30, 40))
    with pytest.raises(ValueError, match="At least one model to fit"):
        fit_sqdt_models([], measurements, fixed_models=[TRUE_NP])
    with pytest.raises(TypeError, match="must be ModelSQDTFit"):
        fit_sqdt_models([TRUE_NP], measurements)  # type: ignore [list-item]
    with pytest.raises(ValueError, match="same species"):
        fit_sqdt_models([ModelSQDTFit("Rb", "np_3/2", [2.64])], measurements, fixed_models=[TRUE_NP])
    with pytest.raises(ValueError, match="exactly one of the given models"):
        fit_sqdt_models([ModelSQDTFit("Cs", "ns_1/2", [4.0])], measurements)
    with pytest.raises(ValueError, match="free offset"):
        fit_sqdt_models([ModelSQDTFit("Cs", "np_3/2", [3.5])], measurements, free_offsets=["unknown"])
    intervals = [
        Measurement(SQDTLevel("np_3/2", n + 1), 1.0, SIGMA_AU, lower=SQDTLevel("np_3/2", n), dataset="intervals")
        for n in range(30, 40)
    ]
    with pytest.raises(ValueError, match=r"do not determine the parameters \['offset intervals'\]"):
        fit_sqdt_models([ModelSQDTFit("Cs", "np_3/2", [3.5])], intervals, free_offsets=["intervals"])
    with pytest.raises(ValueError, match="must be positive"):
        Measurement(SQDTLevel("np_3/2", 30), 0.1, 0)


def test_fitted_models_can_be_fixed_in_a_second_fit() -> None:
    """Models returned by a fit are only fitted again if they are passed as models, not as fixed_models."""
    first = fit_sqdt_models(
        [ModelSQDTFit("Cs", "np_3/2", [3.55, 0.3])], make_level_measurements(TRUE_NP, range(30, 80, 3))
    )
    fitted_np = first.models[0]
    measurements = [
        Measurement(
            SQDTLevel("ns_1/2", n + 4),
            TRUE_NS.calc_energy_au(n=n + 4) - fitted_np.calc_energy_au(n=n),
            SIGMA_AU,
            lower=SQDTLevel("np_3/2", n),
        )
        for n in range(40, 90, 5)
    ]
    second = fit_sqdt_models([ModelSQDTFit("Cs", "ns_1/2", [4.0, 0.2])], measurements, fixed_models=[fitted_np])
    assert len(second.parameter_names) == 2
    assert second.models[0].name == "ns_1/2"


def test_level_str() -> None:
    assert str(SQDTLevel("np_3/2", 45)) == "45p_3/2"


def test_covariance_of_badly_scaled_jacobian() -> None:
    """The covariance is correct even if the columns of the Jacobian differ by many orders of magnitude."""
    rng = np.random.default_rng(1)
    well_conditioned = rng.normal(size=(20, 3))
    scales = np.array([1e8, 1.0, 1e-6])
    expected = np.linalg.inv(well_conditioned.T @ well_conditioned) / np.outer(scales, scales)
    covariance = _calc_covariance(well_conditioned * scales)
    assert np.allclose(covariance, expected, rtol=1e-8, atol=0)


def test_high_order_defect_of_a_low_l_series() -> None:
    """Fit small high order terms, whose finite difference derivatives of absolute energies would vanish.

    A step of delta_4 changes the absolute energies (about 0.14 hartree) by less than their floating point
    resolution, so a finite difference Jacobian would have a zero column; the analytic Jacobian does not.
    """
    true_model = ModelSQDT("Cs", "nf_5/2", [0.03341493, -0.20036, 0.2825])
    measurements = make_level_measurements(true_model, range(28, 69))
    result = fit_sqdt_models([ModelSQDTFit("Cs", "nf_5/2", [0.0334, -0.19, 0.25])], measurements)
    assert np.all(np.isfinite(result.covariance))
    assert_fitted(result, true_model)


class ModelSQDTNFit(ModelSQDTFit):
    """delta(n) = d0 + d2 / n^2, i.e. a parametrization in n instead of n - d0 (only for the tests)."""

    def calc_nui(self, n: int) -> float:
        d0, d2 = self.quantum_defects
        return n - d0 - d2 / n**2

    def calc_nui_gradient(self, n: int) -> list[float]:
        return [-1.0, -1.0 / n**2]

    def to_model(self, fit_info: Mapping[str, Any] | None = None) -> ModelSQDTNFit:
        return ModelSQDTNFit(
            self.species,
            self.channel,
            self.quantum_defects,
            reference=self.reference,
            n_range=self.n_range,
            fit_info=fit_info,
        )


def test_fit_keeps_the_subclass() -> None:
    """A subclass of ModelSQDTFit is fitted with its own parametrization and returned by its to_model."""
    true_model = ModelSQDTNFit("Cs", "np_3/2", [3.559, 0.4])
    measurements = make_level_measurements(true_model, range(12, 40))
    result = fit_sqdt_models([ModelSQDTNFit("Cs", "np_3/2", [3.5, 0.3])], measurements)
    assert type(result.models[0]) is ModelSQDTNFit
    assert result.chi2 < 1e-6
    assert_fitted(result, true_model)
    ritz = fit_sqdt_models([ModelSQDTFit("Cs", "np_3/2", [3.5, 0.3])], measurements)  # the data are not Ritz-like
    assert ritz.chi2 > 1e3


class ModelSQDTFixedD2Fit(ModelSQDTFit):
    """Only delta_0 is a parameter, delta_2 is fixed, but to_model returns both (only for the tests)."""

    @property
    def number_of_parameters(self) -> int:
        return 1

    def get_parameters(self) -> list[float]:
        return [self.quantum_defects[0]]

    def set_parameters(self, parameters: Sequence[float]) -> None:
        self.quantum_defects = (parameters[0], *self.quantum_defects[1:])

    def calc_nui_gradient(self, n: int) -> list[float]:
        return super().calc_nui_gradient(n)[:1]


def test_covariance_of_a_model_with_another_number_of_parameters() -> None:
    """The covariance blocks are given by the number of parameters, not by the quantum defects of to_model."""
    levels = make_level_measurements(TRUE_NP, range(30, 80, 3))
    intervals = [
        Measurement(
            SQDTLevel("ns_1/2", n + 4),
            TRUE_NS.calc_energy_au(n=n + 4) - TRUE_NP.calc_energy_au(n=n),
            SIGMA_AU,
            lower=SQDTLevel("np_3/2", n),
        )
        for n in range(40, 90, 5)
    ]
    start = [ModelSQDTFixedD2Fit("Cs", "np_3/2", [3.55, 0.37469]), ModelSQDTFit("Cs", "ns_1/2", [4.0, 0.2])]
    result = fit_sqdt_models(start, levels + intervals)

    assert len(result.models[0].quantum_defects) == 2
    assert result.covariance.shape == (3, 3)
    assert np.array_equal(result.get_quantum_defect_covariance("np_3/2"), result.covariance[:1, :1])
    assert np.array_equal(result.get_quantum_defect_covariance("ns_1/2"), result.covariance[1:, 1:])
