from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class SQDTYtterbium174(ModelSetSQDT):
    species = "Yb174"
    is_default = True
    nist_data_file = "nist_data.txt"

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
    models: ClassVar = [
        # singlet
        # 6s ns 1S0: S2
        ModelSQDT(
            "Yb174", "6s ns 1S0", [4 + 0.355101645, 0.277673956, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s np 1P1: S4
        ModelSQDT("Yb174", "6s np 1P1", [3 + 0.922709076, 2.60055203, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"),
        # 6s nd 1D2: S6
        ModelSQDT(
            "Yb174", "6s nd 1D2", [2 + 0.729513646, -0.0377841183, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s nf 1F3: S7
        ModelSQDT("Yb174", "6s nf 1F3", [0.276158949, -12.7258012, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"),
        # triplet
        # 6s ns 3S1: S2  # but adjusted for non-Rydberg Ritz
        ModelSQDT("Yb174", "6s ns 3S1", [4 + 0.4382, 4, -1e4, 8e6, -3e9], reference="Wilson 2019, https://arxiv.org/abs/1912.08754"),
        # 6s np 3P0: S3
        ModelSQDT(
            "Yb174", "6s np 3P0", [3 + 0.953661478, -0.287531374, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s np 3P1: S4
        ModelSQDT(
            "Yb174", "6s np 3P1", [3 + 0.982084772, -5.45063476, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s np 3P2: S5
        ModelSQDT(
            "Yb174", "6s np 3P2", [3 + 0.925150932, -2.69197178, 66.7159709, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s nd 3D1: Table I
        ModelSQDT("Yb174", "6s nd 3D1", [2 + 0.75258093, 0.3826, -483.1, 0.0, 0.0], reference="Peper 2024, http://arxiv.org/abs/2406.01482"),
        # 6s nd 3D2: S6
        ModelSQDT(
            "Yb174", "6s nd 3D2", [2 + 0.752292223, 0.104072325, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s nd 3D3: S1
        ModelSQDT(
            "Yb174", "6s nd 3D3", [2 + 0.72902016, -0.705328923, 829.238844, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s nf 3F2: S1
        ModelSQDT(
            "Yb174", "6s nf 3F2", [0.0718252326, -1.00091963, -106.291066, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"
        ),
        # 6s nf 3F3: S7
        ModelSQDT("Yb174", "6s nf 3F3", [0.0715123712, -0.768462937, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"),
        # 6s nf 3F4: S1
        ModelSQDT("Yb174", "6s nf 3F4", [0.0839027969, -2.91009023, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"),
        # 6s ng 3G3: S1
        ModelSQDT("Yb174", "6s ng 3G3", [0.0260964574, -0.14139526, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"),
        # 6s ng 3G5: S1  # actually from Yb171
        ModelSQDT("Yb174", "6s ng 3G5", [0.02529201, -0.11588052, 0.0, 0.0, 0.0], reference="Kuroda 2025, https://arxiv.org/abs/2507.11487"),
        #  +/- states
        # (4, 4.0, "+"): (0.0262659964, 0.0254568575, 0.0, 0.0, 0.0),  # [3] S8
        # (4, 4.0, "-"): (-0.148808463, -0.134219071, 0.0, 0.0, 0.0),  # [3] S8
    ]
