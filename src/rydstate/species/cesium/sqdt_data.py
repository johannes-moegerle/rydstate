from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTCesium(SQDT):
    species = "Cs"
    is_default = True
    nist_data_file = "nist_data.txt"

    # Phys. Rev. A 93, 013424 (2016) (https://doi.org/10.1103/PhysRevA.93.013424),
    # as corrected by the Erratum Phys. Rev. A 112, 049902 (2025) (https://doi.org/10.1103/5v5y-8m53)
    # I = 31406.4677482(10)stat(34)syst 1/cm.
    ionization_energy = (31406.4677482, "1/cm")

    # -- [1] Phys. Rev. A 93, 013424 (2016)
    # -- [2] Phys. Rev. A 26, 2733 (1982)
    # -- [3] Phys. Rev. A 35, 4650 (1987)
    quantum_defects: ClassVar = {
        (0, 0.5, 1 / 2): [4.0493532, 0.2391, 0.06, 11, -209],  # [1]
        (1, 0.5, 1 / 2): [3.5915871, 0.36273, 0.0, 0.0, 0.0],  # [1]
        (1, 1.5, 1 / 2): [3.5590676, 0.37469, 0.0, 0.0, 0.0],  # [1]
        (2, 1.5, 1 / 2): [2.475365, 0.5554, 0.0, 0.0, 0.0],  # [2]
        (2, 2.5, 1 / 2): [2.4663144, 0.01381, -0.392, -1.9, 0.0],  # [1]
        (3, 2.5, 1 / 2): [0.03341424, -0.198674, 0.28953, -0.2601, 0.0],  # [3]
        (3, 3.5, 1 / 2): [0.033537, -0.191, 0.0, 0.0, 0.0],  # [2]
        (4, 3.5, 1 / 2): [0.00703865, -0.049252, 0.01291, 0.0, 0.0],  # [3]
    }
