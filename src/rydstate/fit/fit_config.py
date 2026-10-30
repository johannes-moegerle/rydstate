r"""Fit the SQDT models of a species as described by a fit config file, e.g. from the terminal.

.. code-block:: bash

    fit_sqdt fits/cesium/fit_config.json            # fit and write the models and the report
    fit_sqdt fits/cesium/fit_config.json --dry-run  # only fit and print the result
    fit_sqdt fits/cesium/fit_config.json --check    # check that the written models are reproduced

A fit config is a json file with the keys (paths are relative to the config file)

- "species": the species, e.g. "Cs",
- "data_directory": the directory with one subdirectory per dataset, see :func:`~rydstate.fit.load_measurements`,
- "models_directory": the directory the fitted models are written to, one json file per model
  (e.g. "np_3_2.json", see :meth:`~rydstate.qdt.ModelSQDT.to_dict`), usually the sqdt_models of the species,
- "report_file": the json file the fit report is written to (fit statistics, models, offsets and covariance;
  the values with large residuals are printed by the fit_sqdt script),
- "series": the models to fit, by their channel label (e.g. "np_3/2"), each with the keys

  - "orders": the number of quantum defect coefficients (delta_0, delta_2, ...),
  - "n_min": the smallest n of the model, i.e. where it takes over from the NIST levels of the channel;
    the fit only uses the measured levels from n_min on, usually n_min is the lowest measured level,
  - "start" (optional): start values of the coefficients, padded with zeros, default: all zeros
    (so the fit does not depend on the models it writes),

- "datasets": the datasets to fit, by their directory name, each with the optional keys

  - "free_offset": whether the energies of the dataset have a free offset, i.e. their own ionization threshold,
    see :func:`~rydstate.fit.fit_sqdt_models` (default false),
  - "systematic": {"value": ..., "unit": ...} added in quadrature to the uncertainty of every value,
  - "energies", "intervals": whether energies from the ground state, and intervals, are used (default true),
  - "n_min": the smallest n of the levels that are used,
  - "channels": the channels of the levels that are used (default: all fitted series),
  - "exclude": {source_label: reason} of values that are not used,

and optionally "comment" keys (for the reasons of the choices), which are ignored.
Measurements of levels without a fitted series are not used.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from rydstate import units
from rydstate.fit.fit_sqdt import fit_sqdt_models
from rydstate.fit.measurements import load_measurements
from rydstate.qdt import ModelSQDT, ModelSQDTFit

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from os import PathLike

    from rydstate.fit.fit_sqdt import FitResult, Measurement

CONFIG_KEYS = {"species", "data_directory", "models_directory", "report_file", "series", "datasets", "comment"}
SERIES_KEYS = {"orders", "n_min", "start", "comment"}
DATASET_KEYS = {"free_offset", "systematic", "energies", "intervals", "n_min", "channels", "exclude", "comment"}


@dataclasses.dataclass
class ConfigFit:
    """Result of :func:`fit_from_config`."""

    config_file: Path
    config: dict[str, Any]
    result: FitResult
    """The fit result, whose models are the models as written (with their n range, references and fit info)."""
    datasets: dict[str, dict[str, Any]]
    """For every dataset the number of loaded and used measurements and why the others were not used."""

    @property
    def models_directory(self) -> Path:
        return Path(os.path.normpath(self.config_file.parent / str(self.config["models_directory"])))

    @property
    def report_file(self) -> Path:
        return Path(os.path.normpath(self.config_file.parent / str(self.config["report_file"])))


def fit_from_config(config_file: str | PathLike[str]) -> ConfigFit:
    """Fit the SQDT models as described by a fit config file, see :mod:`rydstate.fit.fit_config`.

    Args:
        config_file: The fit config file.

    Returns:
        The fit, whose models are the models to be written by :func:`write_fit`.

    """
    config_file = Path(config_file)
    config = json.loads(config_file.read_text())
    _check_keys(
        config, CONFIG_KEYS, {"species", "data_directory", "models_directory", "series", "datasets"}, str(config_file)
    )
    species = config["species"]
    series: dict[str, dict[str, Any]] = config["series"]
    for name, options in series.items():
        _check_keys(options, SERIES_KEYS, {"orders", "n_min"}, f"{config_file} series {name}")

    data_directory = config_file.parent / config["data_directory"]
    measurements: list[Measurement] = []
    datasets: dict[str, dict[str, Any]] = {}
    for dataset, options in config["datasets"].items():
        _check_keys(options, DATASET_KEYS, set(), f"{config_file} dataset {dataset}")
        selected, datasets[dataset] = _select(load_measurements(data_directory / dataset), options, series)
        measurements += selected
    free_offsets = [dataset for dataset, options in config["datasets"].items() if options.get("free_offset", False)]

    start = []
    for name, options in series.items():
        values = options.get("start", [])
        if len(values) > options["orders"]:
            raise ValueError(f"{config_file} series {name}: more start values than orders.")
        values = [*values, *[0.0] * (options["orders"] - len(values))]
        model = ModelSQDTFit(species, name, values, n_range=(options["n_min"], math.inf))
        if model.name != name:
            raise ValueError(f"{config_file}: use the label {model.name!r} instead of {name!r} for the series.")
        start.append(model)

    result = fit_sqdt_models(start, measurements, free_offsets=free_offsets)
    for fitted in result.models:
        fitted.fit_info = {**(fitted.fit_info or {}), "config": _config_name(config_file)}
    return ConfigFit(config_file, config, result, datasets)


def write_fit(fit: ConfigFit) -> list[Path]:
    """Write the fitted models (one json file per model) and the fit report of :func:`fit_from_config`.

    Returns:
        The written files.

    """
    fit.models_directory.mkdir(parents=True, exist_ok=True)
    files = []
    for model in fit.result.models:
        file = fit.models_directory / _model_file_name(model.name)
        file.write_text(_dumps(dict(model.to_dict())))
        files.append(file)
    if "report_file" in fit.config:
        fit.report_file.parent.mkdir(parents=True, exist_ok=True)
        fit.report_file.write_text(_dumps(_report(fit)))
        files.append(fit.report_file)
    return files


def check_fit(fit: ConfigFit, tolerance: float = 1e-3) -> list[str]:
    """Compare the fitted models with the written ones.

    Args:
        fit: The fit, see :func:`fit_from_config`.
        tolerance: The allowed difference of the quantum defects in units of their (unscaled) standard deviations,
            see :meth:`~rydstate.fit.FitResult.get_quantum_defect_covariance`.

    Returns:
        A description of every difference (an empty list if the written models are reproduced).

    """
    differences = []
    for model in fit.result.models:
        file = fit.models_directory / _model_file_name(model.name)
        if not file.exists():
            differences.append(f"{file} does not exist")
            continue
        written = ModelSQDT.from_dict(json.loads(file.read_text()))
        sigmas = np.sqrt(np.diag(fit.result.get_quantum_defect_covariance(model.name)))
        if len(written.quantum_defects) != len(model.quantum_defects) or any(
            abs(a - b) > tolerance * sigma
            for a, b, sigma in zip(written.quantum_defects, model.quantum_defects, sigmas, strict=True)
        ):
            differences.append(f"{file}: {written.quantum_defects} written, {model.quantum_defects} fitted")
        if written.n_range != model.n_range or written.reference != model.reference:
            differences.append(f"{file}: n_range or reference differ")
    return differences


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point of the ``fit_sqdt`` script, see :mod:`rydstate.fit.fit_config`."""
    parser = argparse.ArgumentParser(description="Fit the SQDT models of a species as described by fit config files.")
    parser.add_argument("config", nargs="+", help="fit config files, e.g. fits/cesium/fit_config.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", help="only fit and print, do not write any files")
    group.add_argument("--check", action="store_true", help="check that the written models are reproduced")
    args = parser.parse_args(argv)

    failed = False
    for config in args.config:
        fit = fit_from_config(config)
        print(_summary(fit))  # noqa: T201
        if args.check:
            differences = check_fit(fit)
            print("\n".join(differences) if differences else "the written models are reproduced")  # noqa: T201
            failed |= bool(differences)
        elif not args.dry_run:
            for file in write_fit(fit):
                print(f"written: {file}")  # noqa: T201
    if failed:
        raise SystemExit(1)


def _check_keys(data: dict[str, Any], allowed: set[str], required: set[str], where: str) -> None:
    if unknown := set(data) - allowed:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}, allowed are {sorted(allowed)}.")
    if missing := required - set(data):
        raise ValueError(f"{where}: missing keys {sorted(missing)}.")


def _select(
    measurements: list[Measurement], options: dict[str, Any], series: dict[str, Any]
) -> tuple[list[Measurement], dict[str, Any]]:
    """Return the measurements of a dataset that are used, and how many were not used for which reason."""
    exclude: dict[str, str] = options.get("exclude", {})
    if missing := set(exclude) - {m.label for m in measurements}:
        raise ValueError(f"The excluded values {sorted(missing)} are not in the dataset.")
    channels = set(options.get("channels", series))
    reasons: dict[str, Callable[[Measurement], bool]] = {
        "excluded": lambda m: m.label in exclude,
        "energies not used": lambda m: m.lower is None and not options.get("energies", True),
        "intervals not used": lambda m: m.lower is not None and not options.get("intervals", True),
        "level without fitted series": lambda m: any(level.channel not in series for level in m.levels),
        "level not in channels": lambda m: any(level.channel not in channels for level in m.levels),
        "level below n_min": lambda m: any(level.n < options.get("n_min", 0) for level in m.levels),
        "level below n_min of its series": lambda m: any(
            level.n < series[level.channel]["n_min"] for level in m.levels
        ),
    }
    summary: dict[str, Any] = {"loaded": len(measurements), "used": 0}
    selected = []
    for m in measurements:
        reason = next((reason for reason, applies in reasons.items() if applies(m)), None)
        if reason is not None:
            summary[reason] = summary.get(reason, 0) + 1
            continue
        if "systematic" in options:
            systematic_au = units.user_to_au(options["systematic"]["value"], options["systematic"]["unit"], "energy")
            m = dataclasses.replace(m, uncertainty_au=math.hypot(m.uncertainty_au, systematic_au))  # noqa: PLW2901
        selected.append(m)
    summary["used"] = len(selected)
    return selected, summary


def _config_name(config_file: Path) -> str:
    """Return the config file with the names of the two directories above it, e.g. "fits/cesium/fit_config.json"."""
    directory = config_file.resolve().parent
    return f"{directory.parent.name}/{directory.name}/{config_file.name}"


def _model_file_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_") + ".json"


def _report(fit: ConfigFit) -> dict[str, Any]:
    result = fit.result
    return {
        "config": _config_name(fit.config_file),
        "species": fit.config["species"],
        "success": result.success,
        "message": result.message,
        "chi2": result.chi2,
        "dof": result.dof,
        "reduced_chi2": result.reduced_chi2,
        "datasets": fit.datasets,
        "models": [model.to_dict() for model in result.models],
        "offsets_au": result.offsets_au,
        "offset_uncertainties_au": result.offset_uncertainties_au,
        "parameter_names": result.parameter_names,
        "covariance": result.covariance.tolist(),
    }


def _summary(fit: ConfigFit, outlier_sigmas: float = 3) -> str:
    """Return the fit statistics, quantum defects, offsets and the values with a residual above outlier_sigmas."""
    result = fit.result
    scale = np.sqrt(result.reduced_chi2)
    mhz = units.au_to_user(1.0, "energy", "MHz")
    lines = [
        f"{fit.config_file}: chi2/dof = {result.chi2:.1f}/{result.dof} = {result.reduced_chi2:.3f} ({result.message})",
        "quantum defects (uncertainties scaled by sqrt(chi2/dof)):",
    ]
    for model in result.models:
        sigmas = np.sqrt(np.diag(result.get_quantum_defect_covariance(model.name)))
        values = "  ".join(
            f"{value:.9g} +- {sigma * scale:.2g}" for value, sigma in zip(model.quantum_defects, sigmas, strict=True)
        )
        lines.append(f"  {model.name:10s} {values}")
    lines.append("offsets (dataset threshold - rydstate threshold = -offset):")
    for dataset, offset in result.offsets_au.items():
        sigma = result.offset_uncertainties_au[dataset] * scale
        lines.append(f"  {dataset:20s} {-offset * mhz * 1e3:10.1f} +- {sigma * mhz * 1e3:.1f} kHz")
    outliers = sorted(
        (
            (residual / m.uncertainty_au, residual, m)
            for m, residual in zip(result.measurements, result.residuals_au.tolist(), strict=True)
            if abs(residual) > outlier_sigmas * m.uncertainty_au
        ),
        key=lambda outlier: -abs(outlier[0]),
    )
    lines.append(f"{len(outliers)} values with abs(residual) > {outlier_sigmas:g} sigma (residual = fit - measured):")
    lines += [
        f"  {m.dataset:20s} {m.label:24s} {z:+6.1f} sigma {residual * mhz * 1e3:+10.1f} kHz"
        for z, residual, m in outliers
    ]
    return "\n".join(lines)


def _dumps(data: dict[str, Any]) -> str:
    """Return the data as indented json, but with lists of numbers or strings on one line."""
    text = json.dumps(data, indent=2, ensure_ascii=False)
    text = re.sub(
        r"\[\n\s+([^\[\]{}]*?)\n\s*\]",
        lambda match: "[" + ", ".join(item.strip() for item in match.group(1).split(",\n")) + "]",
        text,
    )
    return text + "\n"
