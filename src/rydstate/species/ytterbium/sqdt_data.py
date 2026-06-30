from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTYtterbium174(SQDT):
    species = "Yb174"
    is_default = True
    nist_data_file = "nist_data.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440644&Mask=20
    ionization_energy = (6.25416, "eV")

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
    quantum_defects: ClassVar = {
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


class SQDTYtterbium171(SQDT):
    species = "Yb171"
    is_default = True


class SQDTYtterbium173(SQDT):
    species = "Yb173"
    is_default = True
