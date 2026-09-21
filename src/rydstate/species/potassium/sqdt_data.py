from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTPotassium(SQDT):
    species = "K"
    is_default = True
    nist_data_file = "nist_data.txt"

    # Phys. Rev. A 100, 012501 (2019) (https://doi.org/10.1103/PhysRevA.100.012501)
    # Precision measurement of the ionization energy and quantum defects of 39K I
    # I(39K) = 35009.8139710(22)(sys)(3)(stat) 1/cm
    ionization_energy = (35009.813971, "1/cm")

    # -- [1] Phys. Scr. 27, 300 (1983)
    # -- [2] Opt. Commun. 39, 370 (1981)
    # -- [3] Ark. Fys., 10 p.583 (1956)
    quantum_defects: ClassVar = {
        (0, 0.5, 1 / 2): [2.180197, 0.136, 0.0759, 0.117, -0.206],  # [1,2]
        (1, 0.5, 1 / 2): [1.713892, 0.2332, 0.16137, 0.5345, -0.234],  # [1]
        (1, 1.5, 1 / 2): [1.710848, 0.2354, 0.11551, 1.105, -2.0356],  # [1]
        (2, 1.5, 1 / 2): [0.27697, -1.0249, -0.709174, 11.839, -26.689],  # [1,2]
        (2, 2.5, 1 / 2): [0.277158, -1.0256, -0.59201, 10.0053, -19.0244],  # [1,2]
        (3, 2.5, 1 / 2): [0.010098, -0.100224, 1.56334, -12.6851, 0],  # [1,3]
        (3, 3.5, 1 / 2): [0.010098, -0.100224, 1.56334, -12.6851, 0],  # [1,3]
    }
