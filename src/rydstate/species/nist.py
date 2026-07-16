from __future__ import annotations

import inspect
import re
from fractions import Fraction
from pathlib import Path

import numpy as np


# A parsed NIST energy level is keyed by (n, l, j_tot, s_tot) and maps to the level energy in Hartree.
NistEnergyLevels = dict[tuple[int, int, float, float], float]


def resolve_species_data_file(cls: type, filename: str) -> Path:
    """Resolve a data file located next to the module defining ``cls``.

    The species specific classes (e.g. the SQDT and MQDT subclasses) live in the species directory
    together with their data files. This helper returns the absolute path of ``filename`` in that directory.

    Args:
        cls: The class whose defining module directory contains the data file.
        filename: The name of the data file (relative to the species directory).

    Returns:
        The absolute path of the data file.

    """
    return Path(inspect.getfile(cls)).resolve().parent / filename


def parse_nist_energy_levels(  # noqa: C901, PLR0912
    file: Path, core_electron_configuration: str, *, species: str | None = None
) -> NistEnergyLevels:
    """Parse the low-lying NIST energy levels from a NIST data file.

    The file should be directly downloaded from https://physics.nist.gov/PhysRefData/ASD/levels_form.html
    in the 'Tab-delimited' format and in units of Hartree.

    Only single valence electron states (i.e. states whose inner electrons are in the ground state
    configuration of the ionic core) are kept, since only those can be described by the (S)QDT model.

    Args:
        file: The path to the NIST data file.
        core_electron_configuration: The electron configuration of the ionic core (e.g. ``"4f14.6s"``),
            used to identify the single valence electron and its (n, l) quantum numbers.
        species: The species name, only used to make error messages more descriptive.

    Returns:
        A dictionary mapping (n, l, j_tot, s_tot) to the level energy in Hartree.

    """
    if not file.exists():
        raise ValueError(f"NIST energy data file {file} does not exist.")

    header = file.read_text().splitlines()[0]
    if "Level (Hartree)" not in header:
        raise ValueError(
            f"NIST energy data file {file} not given in Hartree, please download the data in units of Hartree."
        )

    data = np.loadtxt(file, skiprows=1, dtype=str, quotechar='"', delimiter="\t")
    # data[i] := (Configuration, Term, J, Prefix, Energy, Suffix, Uncertainty, Reference)
    core_config_parts = convert_electron_configuration(core_electron_configuration)

    nist_energy_levels: NistEnergyLevels = {}
    for row in data:
        if re.match(r"^([A-Z])", row[0]):
            # Skip rows, where the first column starts with an element symbol
            continue

        try:
            config_parts = convert_electron_configuration(row[0])
        except ValueError:
            # Skip rows with invalid electron configuration format
            # (they usually correspond to core configurations, that are not the ground state configuration)
            # e.g. strontium "4d.(2D<3/2>).4f"
            continue
        if sum(part[2] for part in config_parts) != sum(part[2] for part in core_config_parts) + 1:
            # Skip configurations, where the number of electrons does not match the core configuration + 1
            continue

        for part in core_config_parts:
            if part in config_parts:
                config_parts.remove(part)
            elif (part[0], part[1], part[2] + 1) in config_parts:
                config_parts.remove((part[0], part[1], part[2] + 1))
                config_parts.append((part[0], part[1], 1))
            else:
                break
        if sum(part[2] for part in config_parts) != 1:
            # Skip configurations, where the inner electrons are not in the ground state configuration
            continue
        n, l = config_parts[0][:2]

        multiplicity = int(row[1][0])
        s_tot = (multiplicity - 1) / 2

        j_tot_list = [float(Fraction(j_str)) for j_str in row[2].split(",")]
        for j_tot in j_tot_list:
            energy = float(row[4])
            nist_energy_levels[(n, l, j_tot, s_tot)] = energy

    if len(nist_energy_levels) == 0:
        raise ValueError(f"No NIST energy levels found for species {species} in file {file}.")

    return nist_energy_levels


def convert_electron_configuration(config: str) -> list[tuple[int, int, int]]:
    """Convert an electron configuration string to a list of tuples [(n, l, number), ...].

    This means convert a string representing the outermost electrons
    like "4f14.6s" to [(4, 3, 14), (6, 0, 1)].
    """
    l_str2int = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5, "i": 6, "k": 7, "l": 8, "m": 9}
    parts = config.split(".")
    converted_parts = []
    for part in parts:
        match = re.match(r"^(\d+)([a-z])(\d*)$", part)
        if match is None:
            raise ValueError(f"Invalid configuration format: {config}.")
        n = int(match.group(1))
        l = l_str2int[match.group(2)]
        number = int(match.group(3)) if match.group(3) else 1
        converted_parts.append((n, l, number))

    return converted_parts
