from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from rydstate.species.sqdt.species_object_sqdt import SpeciesObjectSQDT
from rydstate.units import electron_mass, rydberg_constant


class _YtterbiumAbstract(SpeciesObjectSQDT):
    Z = 70
    number_valence_electrons = 2
    ground_state_shell = (6, 0)
    _additional_allowed_shells: ClassVar = [(5, 2), (5, 3), (5, 4)]

    _core_electron_configuration = "4f14.6s"
    _nist_energy_levels_file = Path(__file__).parent / "nist_energy_levels" / "ytterbium.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440644&Mask=20
    _ionization_energy = (6.25416, None, "eV")

    potential_type_default = "model_potential_fei_2009"

    # https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)


class Ytterbium171(_YtterbiumAbstract):
    name = "Yb171"
    i_c = 1 / 2

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable1.htm
    _isotope_mass = 170.936323  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )


class Ytterbium173(_YtterbiumAbstract):
    name = "Yb173"
    i_c = 5 / 2

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable1.htm
    _isotope_mass = 172.938208  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )


class Ytterbium174(_YtterbiumAbstract):
    name = "Yb174"
    i_c = 0

    # https://physics.nist.gov/PhysRefData/Handbook/Tables/ytterbiumtable1.htm
    _isotope_mass = 173.938859  # u
    _corrected_rydberg_constant = (
        rydberg_constant.m / (1 + electron_mass.to("u").m / _isotope_mass),
        None,
        str(rydberg_constant.u),
    )

    # -- [1] Peper 2024, http://arxiv.org/abs/2406.01482
    #        Spectroscopy and modeling of 171Yb Rydberg states for high-fidelity two-qubit gates
    #        see Table I
    #        Isotope Yb174
    # -- [2] Wilson 2019, https://arxiv.org/abs/1912.08754
    #        Trapped arrays of alkaline earth Rydberg atoms in optical tweezers
    #        fit only valid from 28s upward
    #        see Table S2, but adjusted for non-Rydberg Ritz
    #        Isotope Yb174
    # -- [3] Kuroda 2025, https://arxiv.org/abs/2507.11487
    #        Microwave spectroscopy and multi-channel quantum defect analysis of ytterbium Rydberg states
    #        see Table S1 - S8
    #        Isotope Yb174

    _quantum_defects: ClassVar = {
        # singlet
        (0, 0.0, 0): (4 + 0.355101645, 0.277673956, 0.0, 0.0, 0.0),  # [3] S2
        (1, 1.0, 0): (3 + 0.922709076, 2.60055203, 0.0, 0.0, 0.0),  # [3] S4
        (2, 2.0, 0): (2 + 0.729513646, -0.0377841183, 0.0, 0.0, 0.0),  # [3] S6
        (3, 3.0, 0): (0.276158949, -12.7258012, 0.0, 0.0, 0.0),  # [3] S7
        # triplet
        (0, 1.0, 1): (4 + 0.4382, 4, -1e4, 8e6, -3e9),  # [2] S2  # but adjusted for non-Rydberg Ritz
        (1, 0.0, 1): (3 + 0.953661478, -0.287531374, 0.0, 0.0, 0.0),  # [3] S3
        (1, 1.0, 1): (3 + 0.982084772, -5.45063476, 0.0, 0.0, 0.0),  # [3] S4
        (1, 2.0, 1): (3 + 0.925150932, -2.69197178, 66.7159709, 0.0, 0.0),  # [3] S5
        (2, 1.0, 1): (2 + 0.75258093, 0.3826, -483.1, 0.0, 0.0),  # [1] Table I
        (2, 2.0, 1): (2 + 0.752292223, 0.104072325, 0.0, 0.0, 0.0),  # [3] S6
        (2, 3.0, 1): (2 + 0.72902016, -0.705328923, 829.238844, 0.0, 0.0),  # [3] S1
        (3, 2.0, 1): (0.0718252326, -1.00091963, -106.291066, 0.0, 0.0),  # [3] S1
        (3, 3.0, 1): (0.0715123712, -0.768462937, 0.0, 0.0, 0.0),  # [3] S7
        (3, 4.0, 1): (0.0839027969, -2.91009023, 0.0, 0.0, 0.0),  # [3] S1
        (4, 3.0, 1): (0.0260964574, -0.14139526, 0.0, 0.0, 0.0),  # [3] S1
        (4, 5.0, 1): (0.02529201, -0.11588052, 0.0, 0.0, 0.0),  # [3] S1  # actually from Yb171
        #  +/- states
        # (4, 4.0, "+"): (0.0262659964, 0.0254568575, 0.0, 0.0, 0.0),  # [3] S8
        # (4, 4.0, "-"): (-0.148808463, -0.134219071, 0.0, 0.0, 0.0),  # [3] S8
    }
