from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class SQDTCesium(ModelSetSQDT):
    species = "Cs"
    nist_data_file = "nist_data.txt"

    # -- [1] Phys. Rev. A 93, 013424 (2016)
    # -- [2] Phys. Rev. A 26, 2733 (1982)
    # -- [3] Phys. Rev. A 35, 4650 (1987)
    models: ClassVar = [
        ModelSQDT("Cs", "ns_1/2", [4.0493532, 0.2391, 0.06, 11, -209], reference="Phys. Rev. A 93, 013424 (2016)"),
        ModelSQDT("Cs", "np_1/2", [3.5915871, 0.36273, 0.0, 0.0, 0.0], reference="Phys. Rev. A 93, 013424 (2016)"),
        ModelSQDT("Cs", "np_3/2", [3.5590676, 0.37469, 0.0, 0.0, 0.0], reference="Phys. Rev. A 93, 013424 (2016)"),
        ModelSQDT("Cs", "nd_3/2", [2.475365, 0.5554, 0.0, 0.0, 0.0], reference="Phys. Rev. A 26, 2733 (1982)"),
        ModelSQDT("Cs", "nd_5/2", [2.4663144, 0.01381, -0.392, -1.9, 0.0], reference="Phys. Rev. A 93, 013424 (2016)"),
        ModelSQDT(
            "Cs", "nf_5/2", [0.03341424, -0.198674, 0.28953, -0.2601, 0.0], reference="Phys. Rev. A 35, 4650 (1987)"
        ),
        ModelSQDT("Cs", "nf_7/2", [0.033537, -0.191, 0.0, 0.0, 0.0], reference="Phys. Rev. A 26, 2733 (1982)"),
        ModelSQDT("Cs", "ng_7/2", [0.00703865, -0.049252, 0.01291, 0.0, 0.0], reference="Phys. Rev. A 35, 4650 (1987)"),
    ]
