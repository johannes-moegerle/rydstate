from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class SQDTPotassium(ModelSetSQDT):
    species = "K"
    is_default = True
    nist_data_file = "nist_data.txt"

    # -- [1] Phys. Scr. 27, 300 (1983)
    # -- [2] Opt. Commun. 39, 370 (1981)
    # -- [3] Ark. Fys., 10 p.583 (1956)
    models: ClassVar = [
        ModelSQDT(
            "K",
            "ns_1/2",
            [2.180197, 0.136, 0.0759, 0.117, -0.206],
            reference=("Phys. Scr. 27, 300 (1983)", "Opt. Commun. 39, 370 (1981)"),
        ),
        ModelSQDT("K", "np_1/2", [1.713892, 0.2332, 0.16137, 0.5345, -0.234], reference="Phys. Scr. 27, 300 (1983)"),
        ModelSQDT("K", "np_3/2", [1.710848, 0.2354, 0.11551, 1.105, -2.0356], reference="Phys. Scr. 27, 300 (1983)"),
        ModelSQDT(
            "K",
            "nd_3/2",
            [0.27697, -1.0249, -0.709174, 11.839, -26.689],
            reference=("Phys. Scr. 27, 300 (1983)", "Opt. Commun. 39, 370 (1981)"),
        ),
        ModelSQDT(
            "K",
            "nd_5/2",
            [0.277158, -1.0256, -0.59201, 10.0053, -19.0244],
            reference=("Phys. Scr. 27, 300 (1983)", "Opt. Commun. 39, 370 (1981)"),
        ),
        ModelSQDT(
            "K",
            "nf_5/2",
            [0.010098, -0.100224, 1.56334, -12.6851, 0],
            reference=("Phys. Scr. 27, 300 (1983)", "Ark. Fys., 10 p.583 (1956)"),
        ),
        ModelSQDT(
            "K",
            "nf_7/2",
            [0.010098, -0.100224, 1.56334, -12.6851, 0],
            reference=("Phys. Scr. 27, 300 (1983)", "Ark. Fys., 10 p.583 (1956)"),
        ),
    ]
