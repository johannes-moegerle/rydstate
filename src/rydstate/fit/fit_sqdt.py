from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import least_squares

from rydstate.angular.angular_ket import AngularKetBase, AngularKetLS
from rydstate.angular.short_label import format_short_label, parse_short_label
from rydstate.qdt.sqdt.model_sqdt_fit import ModelSQDTFit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rydstate.qdt.sqdt.model_sqdt import ModelSQDT, ModelSQDTBase
    from rydstate.units import NDArray


@dataclass(frozen=True)
class SQDTLevel:
    """A Rydberg level, identified by the short label of its channel (e.g. "np_3/2") and its principal quantum number.

    The channel label must be the name of the SQDT model describing the level, see :class:`~rydstate.qdt.ModelSQDT`.
    """

    channel: str
    n: int

    def __str__(self) -> str:
        return self.channel.replace("n", str(self.n), 1)

    @classmethod
    def from_label(cls, label: str, species: str) -> SQDTLevel:
        """Create a level from its short label including n, e.g. "45p_3/2" or "5s 40d 1D2".

        The label is the short label of the channel (see
        :meth:`~rydstate.angular.angular_ket.AngularKetBase.get_short_label`) with the principal quantum number n
        in place of the placeholder "n". The channel is stored with its canonical short label.

        Args:
            label: The short label of the level.
            species: The species, which determines the format of the label (single or two valence electrons).

        """
        for template in (AngularKetLS.short_label_template_single_valence, AngularKetLS.short_label_template):
            try:
                values = parse_short_label(template, label)
            except ValueError:
                continue
            if "n" in values:
                channel = AngularKetBase.from_short_label(format_short_label(template, {**values, "n": None}), species)
                return cls(channel.get_short_label(), int(values["n"]))
        raise ValueError(f"The label {label!r} is no short label of a level with principal quantum number n.")


@dataclass(frozen=True)
class Measurement:
    """A measured energy of a Rydberg level, or a measured interval between two Rydberg levels.

    The measured value is E(upper) if lower is None, i.e. the energy of the level measured from the ground state
    (the zero of the energies of rydstate), otherwise the interval E(upper) - E(lower).
    """

    upper: SQDTLevel
    value_au: float
    """Measured energy or interval in atomic units (Hartree)."""
    uncertainty_au: float
    """Uncertainty (standard deviation) of the measured value in atomic units (Hartree)."""
    lower: SQDTLevel | None = None
    dataset: str = ""
    """Name of the dataset the measurement belongs to, used for free offsets, see :func:`fit_sqdt_models`."""
    label: str = ""
    reference: str = ""
    """The publication of the measured value, e.g. "Shen 2024, Phys. Rev. Lett. 133, 233005, https://doi.org/..."."""

    def __post_init__(self) -> None:
        if not self.uncertainty_au > 0:
            raise ValueError(f"The uncertainty of the measurement {self} must be positive.")

    @property
    def levels(self) -> tuple[SQDTLevel, ...]:
        """The measured level, or the upper and the lower level of an interval."""
        return (self.upper,) if self.lower is None else (self.upper, self.lower)


@dataclass
class FitResult:
    """Result of :func:`fit_sqdt_models`."""

    models: list[ModelSQDT]
    """The fitted models in the order they were given, with their fit info, see :func:`fit_sqdt_models`.
    Their uncertainties are given by :attr:`covariance`, see :meth:`get_quantum_defect_covariance`."""
    offsets_au: dict[str, float]
    """Fitted offset of each dataset with a free offset, see :func:`fit_sqdt_models`."""
    offset_uncertainties_au: dict[str, float]
    """Standard deviation of each fitted offset from :attr:`covariance` (unscaled)."""
    covariance: NDArray
    """Covariance matrix of all fitted parameters, (J^T J)^-1 with the Jacobian J of the weighted residuals,
    including the correlations between different models and the offsets (see :attr:`parameter_names`).
    It assumes that the given uncertainties are correct, scale it by :attr:`reduced_chi2` otherwise."""
    parameter_names: list[str]
    """Names of the rows and columns of :attr:`covariance`: the quantum defects of the fitted models
    ("<species> <model name>: quantum_defects[i]", in the order of the models), followed by the free offsets
    ("offset <dataset>")."""
    number_of_parameters: list[int]
    """Number of fitted parameters of each model (in the order of :attr:`models`),
    see :attr:`~rydstate.qdt.ModelSQDTFit.number_of_parameters`."""
    measurements: list[Measurement]
    residuals_au: NDArray
    """Predicted minus (measured plus offset) value of every measurement in atomic units (Hartree)."""
    chi2: float
    dof: int
    success: bool
    message: str

    @property
    def normalized_residuals(self) -> NDArray:
        """Residuals divided by the uncertainties of the measurements."""
        sigmas = np.array([measurement.uncertainty_au for measurement in self.measurements])
        normalized_residuals: NDArray = self.residuals_au / sigmas
        return normalized_residuals

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else math.nan

    def get_quantum_defect_covariance(self, name: str) -> NDArray:
        """Return the block of :attr:`covariance` of the quantum defects of one fitted model.

        The quantum defects of a model are strongly correlated, so the uncertainty of a quantity derived from them
        (e.g. of an energy) must be calculated with the covariance, not from the standard deviations of its diagonal.
        The block lacks the correlations with the other models and the offsets, which are needed e.g. for an interval
        between levels of different models, see :attr:`covariance`.

        For a model with another parametrization (see :meth:`~rydstate.qdt.ModelSQDTFit.to_model`),
        this is the block of its fitted parameters.

        Args:
            name: The name of the model (e.g. "np_3/2"), see :attr:`~rydstate.qdt.ModelSQDT.name`.

        """
        matching = [index for index, model in enumerate(self.models) if model.name == name]
        if len(matching) != 1:
            raise ValueError(f"{len(matching)} fitted models have the name {name!r}.")
        start = sum(self.number_of_parameters[: matching[0]])
        stop = start + self.number_of_parameters[matching[0]]
        block: NDArray = self.covariance[start:stop, start:stop]
        return block


def fit_sqdt_models(
    models: Sequence[ModelSQDTFit],
    measurements: Sequence[Measurement],
    *,
    fixed_models: Sequence[ModelSQDTBase] = (),
    free_offsets: Sequence[str] = (),
    **least_squares_kwargs: Any,  # noqa: ANN401
) -> FitResult:
    """Fit the quantum defects of SQDT models to measured energies and intervals of Rydberg levels.

    Every level is described by the model of its channel (models and fixed_models), whose n range contains the level.
    The free parameters are the quantum defects of the models (every given order, see
    :meth:`~rydstate.qdt.ModelSQDTFit.get_parameters`) and an offset o for every dataset in free_offsets.
    The given models are the start values, they are not changed (the fit uses copies of them).
    The fitted models are new :class:`~rydstate.qdt.ModelSQDT` models (or what the
    :meth:`~rydstate.qdt.ModelSQDTFit.to_model` of a subclass with another parametrization returns)
    with the n range of the given model and

    - reference: the distinct references of the measurements of levels of the model (see
      :attr:`Measurement.reference`), or None if they have none,
    - fit_info: how the model was fitted (the method, the rydstate version and the n range of the measured levels);
      the quantities of the whole fit (chi2, dof, offsets, datasets, the models fitted together and the fixed models)
      are only in the returned :class:`FitResult` and the arguments.

    The weighted residuals (E_pred - E_meas - o) / sigma are minimized with :func:`scipy.optimize.least_squares`
    and an analytic Jacobian. The offset only applies to energies measured from the ground state (not to intervals),
    so it acts like a free ionization threshold I - o for that dataset: the fitted models then describe the binding
    energies of the data, and are used with the ionization threshold of the species.

    Args:
        models: The models to fit, e.g. ``ModelSQDTFit("Cs", "np_3/2", [3.56, 0.37])`` or a fittable copy of
            a model of a species, see :meth:`~rydstate.qdt.ModelSQDTFit.from_model`.
        measurements: The measured energies and intervals.
        fixed_models: Further models describing measured levels, which are not fitted
            (e.g. the models of the lower levels of intervals, or :class:`~rydstate.qdt.ModelSQDTNist` models).
        free_offsets: Names of the datasets (see :attr:`Measurement.dataset`) with a free offset.
        least_squares_kwargs: Further keyword arguments for :func:`scipy.optimize.least_squares`. The defaults are
            ``x_scale="jac"`` and xtol, ftol, gtol of 1e-14: the scipy default xtol=1e-8 is relative to the
            parameters and stops fits of precise data (a few kHz) far from the minimum.

    Returns:
        The fitted models and parameters, their covariance and the residuals.

    """
    _check_arguments(models, measurements, fixed_models, free_offsets)
    free_models = [copy.copy(model) for model in models]
    all_models = [*free_models, *fixed_models]

    level_models = {
        level: _get_model(level, all_models) for measurement in measurements for level in measurement.levels
    }
    # the levels of each measurement with the sign of their energy in the measured value (+ upper, - lower level)
    signed_levels = [list(zip(measurement.levels, (1.0, -1.0), strict=False)) for measurement in measurements]

    # parameter vector: the quantum defects of the free models, followed by the free offsets
    ends = np.cumsum([model.number_of_parameters for model in free_models]).tolist()
    parameter_slices = {
        model: slice(end - model.number_of_parameters, end) for model, end in zip(free_models, ends, strict=True)
    }
    start = ends[-1]
    offset_rows = np.array(
        [row for row, m in enumerate(measurements) if m.lower is None and m.dataset in free_offsets], dtype=int
    )
    offset_columns = np.array(
        [start + list(free_offsets).index(measurements[row].dataset) for row in offset_rows], dtype=int
    )
    sigmas = np.array([measurement.uncertainty_au for measurement in measurements])

    # the part of the residuals that does not depend on the parameters: the ionization thresholds minus the measured
    # value; thresholds and binding energies are summed separately, so that the large thresholds cancel exactly
    # instead of limiting the precision to about 1e-17 hartree
    threshold_differences = np.array(
        [
            sum((sign * level_models[level].ionization_threshold_au for level, sign in levels), -measurement.value_au)
            for measurement, levels in zip(measurements, signed_levels, strict=True)
        ]
    )
    # the levels described by the free models, whose binding energies depend on the parameters
    free_levels = [
        (row, level, model, sign)
        for row, levels in enumerate(signed_levels)
        for level, sign in levels
        if isinstance(model := level_models[level], ModelSQDTFit) and model in parameter_slices
    ]

    def set_parameters(x: NDArray) -> None:
        for model in free_models:
            model.set_parameters(x[parameter_slices[model]].tolist())

    def calc_residuals_au(x: NDArray) -> NDArray:
        set_parameters(x)
        binding_energies = np.array(
            [
                sum(sign * level_models[level].calc_binding_energy_au(n=level.n) for level, sign in levels)
                for levels in signed_levels
            ]
        )
        residuals: NDArray = threshold_differences + binding_energies
        residuals[offset_rows] -= x[offset_columns]
        return residuals

    def calc_weighted_residuals(x: NDArray) -> NDArray:
        weighted_residuals: NDArray = calc_residuals_au(x) / sigmas
        return weighted_residuals

    def calc_weighted_jacobian(x: NDArray) -> NDArray:
        set_parameters(x)
        jacobian = np.zeros((len(measurements), len(x)))
        for row, level, model, sign in free_levels:
            jacobian[row, parameter_slices[model]] += sign * np.array(model.calc_binding_energy_gradient(level.n))
        jacobian[offset_rows, offset_columns] = -1.0
        weighted_jacobian: NDArray = jacobian / sigmas[:, np.newaxis]
        return weighted_jacobian

    options: dict[str, Any] = {"x_scale": "jac", "xtol": 1e-14, "ftol": 1e-14, "gtol": 1e-14} | least_squares_kwargs
    solution = least_squares(
        calc_weighted_residuals,
        x0=np.array([p for model in free_models for p in model.get_parameters()] + [0.0] * len(free_offsets)),
        jac=calc_weighted_jacobian,
        **options,
    )

    residuals_au = calc_residuals_au(solution.x)  # also sets the fitted parameters of the models
    parameter_names = [
        f"{model.full_name}: quantum_defects[{index}]"
        for model in free_models
        for index in range(model.number_of_parameters)
    ] + [f"offset {dataset}" for dataset in free_offsets]
    covariance = _calc_covariance(calc_weighted_jacobian(solution.x), parameter_names)
    chi2 = float(np.sum((residuals_au / sigmas) ** 2))
    dof = len(measurements) - len(solution.x)
    fitted_models = [model.to_model(_get_fit_info(model, measurements, level_models)) for model in free_models]
    for model, fitted in zip(free_models, fitted_models, strict=True):
        fitted.reference = _get_references(model, measurements, level_models)
    return FitResult(
        models=fitted_models,
        offsets_au={dataset: float(solution.x[start + index]) for index, dataset in enumerate(free_offsets)},
        offset_uncertainties_au={
            dataset: math.sqrt(covariance[start + index, start + index]) for index, dataset in enumerate(free_offsets)
        },
        covariance=covariance,
        parameter_names=parameter_names,
        number_of_parameters=[model.number_of_parameters for model in free_models],
        measurements=list(measurements),
        residuals_au=residuals_au,
        chi2=chi2,
        dof=dof,
        success=bool(solution.success),
        message=str(solution.message),
    )


def _check_arguments(
    models: Sequence[ModelSQDTFit],
    measurements: Sequence[Measurement],
    fixed_models: Sequence[ModelSQDTBase],
    free_offsets: Sequence[str],
) -> None:
    """Check the arguments of fit_sqdt_models."""
    if len(models) == 0:
        raise ValueError("At least one model to fit is needed.")
    if not all(isinstance(model, ModelSQDTFit) for model in models):
        raise TypeError("The models to fit must be ModelSQDTFit, see ModelSQDTFit.from_model.")
    if any(model.species != models[0].species for model in [*models, *fixed_models]):
        species = [model.species for model in [*models, *fixed_models]]
        raise ValueError(f"All models must belong to the same species, got {species}.")
    if unknown_datasets := set(free_offsets) - {measurement.dataset for measurement in measurements}:
        raise ValueError(f"No measurements of the datasets {sorted(unknown_datasets)} with a free offset.")


def _get_references(
    model: ModelSQDTFit, measurements: Sequence[Measurement], level_models: dict[SQDTLevel, ModelSQDTBase]
) -> tuple[str, ...] | None:
    """Return the distinct references of the measurements of levels of the model (in order), or None."""
    references = dict.fromkeys(
        m.reference for m in measurements if m.reference and any(level_models[level] is model for level in m.levels)
    )
    return tuple(references) or None


def _get_fit_info(
    model: ModelSQDTFit, measurements: Sequence[Measurement], level_models: dict[SQDTLevel, ModelSQDTBase]
) -> dict[str, Any]:
    """Return how the model was fitted, i.e. the method and the n range of its measured levels."""
    import rydstate  # noqa: PLC0415 (rydstate imports this module)

    ns = [level.n for m in measurements for level in m.levels if level_models[level] is model]
    return {
        "method": "rydstate.fit.fit_sqdt_models",
        "rydstate version": rydstate.__version__,
        "n range of the data": [min(ns), max(ns)] if ns else None,
    }


def _get_model(level: SQDTLevel, models: Sequence[ModelSQDTBase]) -> ModelSQDTBase:
    """Return the model of the channel of the level, whose n range contains the level."""
    candidates = [model for model in models if model.name == level.channel and model.n_min <= level.n <= model.n_max]
    if len(candidates) != 1:
        raise ValueError(
            f"The level {level} must be described by exactly one of the given models, "
            f"but {len(candidates)} models describe it: {candidates}."
        )
    return candidates[0]


def _calc_covariance(jacobian: NDArray, parameter_names: Sequence[str] | None = None) -> NDArray:
    """Return (J^T J)^-1, computed via the singular value decomposition of the column normalized Jacobian.

    The columns of J can differ by many orders of magnitude (e.g. a dataset offset and a high order quantum defect),
    so J^T J itself is too badly conditioned to be inverted directly.

    Raises:
        ValueError: If the data does not determine all parameters (J has not full column rank), e.g. a free offset
            of a dataset without energies, or a series without measurements.

    """
    column_norms = np.linalg.norm(jacobian, axis=0)
    column_norms[column_norms == 0] = 1
    _, singular_values, vh = np.linalg.svd(jacobian / column_norms, full_matrices=False)
    threshold = singular_values.max(initial=0) * max(jacobian.shape) * np.finfo(float).eps
    if len(singular_values) < jacobian.shape[1] or np.any(singular_values <= threshold):
        if parameter_names is None:
            parameter_names = [f"parameter {index}" for index in range(jacobian.shape[1])]
        _, singular_values, vh = np.linalg.svd(jacobian / column_norms, full_matrices=True)
        null_space = vh[np.sum(singular_values > threshold) :]
        undetermined = [
            name for name, weight in zip(parameter_names, np.abs(null_space).max(axis=0), strict=True) if weight > 1e-3
        ]
        raise ValueError(
            f"The measurements do not determine the parameters {undetermined} (or only a combination of them), "
            "e.g. a free offset of a dataset without energies or a series without measurements."
        )
    covariance: NDArray = ((vh.T / singular_values**2) @ vh) / np.outer(column_norms, column_norms)
    return covariance
