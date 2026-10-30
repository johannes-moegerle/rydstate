from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rydstate import units
from rydstate.fit.fit_sqdt import Measurement, SQDTLevel

if TYPE_CHECKING:
    from os import PathLike

MEASUREMENTS_COLUMNS = ("level", "lower_level", "value", "uncertainty", "unit", "source_label")
"""Required columns of measurements.csv, see :func:`load_measurements`."""


def load_measurements(directory: str | PathLike[str]) -> list[Measurement]:
    """Load the measured energies and intervals of a dataset.

    A dataset is a directory with the files ``metadata.json`` and ``measurements.csv``.
    The name of the directory is the name of the dataset (see :attr:`~rydstate.fit.Measurement.dataset`).
    ``metadata.json`` describes the source, at least by the keys "species" and "reference"
    (e.g. "Shen 2024, Phys. Rev. Lett. 133, 233005, https://doi.org/10.1103/PhysRevLett.133.233005"),
    see :func:`load_metadata`; the reference is used as :attr:`~rydstate.fit.Measurement.reference`.
    Every row of ``measurements.csv`` is one measured value with the columns

    - level: the short label of the level including n, e.g. "45p_3/2" or "5s 40d 1D2",
      see :meth:`~rydstate.fit.SQDTLevel.from_label`,
    - lower_level: empty for an energy of the level measured from the ground state
      (the zero of the energies of rydstate, i.e. the hyperfine centre of gravity of the ground state),
      otherwise the short label of the lower level of the measured interval E(level) - E(lower_level),
    - value, uncertainty, unit: the measured value and its uncertainty (standard deviation) in the unit,
      which can be any energy, frequency or wave number unit of :mod:`rydstate.units`, e.g. "1/cm", "GHz", "MHz"
      or "kHz"; wavelengths are not allowed (the uncertainty would not convert like the value, and published
      wavelengths are often air wavelengths), they must be converted to one of these units in the file,
    - source_label: the label of the value in the source, unique within the dataset,
      it is used as :attr:`~rydstate.fit.Measurement.label`,

    and optionally further columns (e.g. source_location and note), which are ignored.
    All corrections of the values (e.g. to the energy zero of rydstate) are already applied in the file.

    Args:
        directory: The directory of the dataset.

    Returns:
        The measurements of the dataset, in the order of the rows.

    """
    directory = Path(directory)
    metadata = load_metadata(directory)
    species = metadata["species"]
    with (directory / "measurements.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        missing = set(MEASUREMENTS_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{directory / 'measurements.csv'}: missing columns {sorted(missing)}.")
        rows = list(reader)
    labels = [row["source_label"] for row in rows]
    if len(set(labels)) != len(labels):
        raise ValueError(f"{directory / 'measurements.csv'}: the source labels are not unique.")

    measurements = []
    for row in rows:
        if not _is_proportional_to_energy(row["unit"]):
            raise ValueError(
                f"{directory / 'measurements.csv'}: the unit {row['unit']!r} of {row['source_label']!r} is not "
                "an energy, frequency or wave number unit, see rydstate.fit.load_measurements."
            )
        lower = SQDTLevel.from_label(row["lower_level"], species) if row["lower_level"] else None
        measurements.append(
            Measurement(
                SQDTLevel.from_label(row["level"], species),
                units.user_to_au(float(row["value"]), row["unit"], "energy"),
                units.user_to_au(float(row["uncertainty"]), row["unit"], "energy"),
                lower=lower,
                dataset=directory.name,
                label=row["source_label"],
                reference=metadata["reference"],
            )
        )
    return measurements


def _is_proportional_to_energy(unit: str) -> bool:
    """Return whether values in the unit are proportional to the energy, so their uncertainties convert likewise."""
    quantity = units.ureg.Quantity(1, unit)
    return any(quantity.check(dimension) for dimension in ("[energy]", "[frequency]", "1/[length]"))


def load_metadata(directory: str | PathLike[str]) -> dict[str, Any]:
    """Load the metadata.json of a dataset, which must contain at least the keys "species" and "reference".

    Args:
        directory: The directory of the dataset, see :func:`load_measurements`.

    """
    path = Path(directory) / "metadata.json"
    metadata: dict[str, Any] = json.loads(path.read_text())
    missing = {"species", "reference"} - set(metadata)
    if missing:
        raise ValueError(f"{path}: missing keys {sorted(missing)}.")
    return metadata
