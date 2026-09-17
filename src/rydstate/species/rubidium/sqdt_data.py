from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTRubidium(SQDT):
    species = "Rb"
    is_default = True
    nist_data_file = "nist_data.txt"

    # https://journals.aps.org/pra/pdf/10.1103/PhysRevA.83.052515
    ionization_energy = (1_010_029.164_6, "GHz")

    # -- [1] Phys. Rev. A 83, 052515 (2011) - Rb87
    # -- [2] Phys. Rev. A 67, 052502 (2003) - Rb85
    # -- [3] Phys. Rev. A 74, 054502 (2006) - Rb85
    # -- [4] Phys. Rev. A 74, 062712 (2006) - Rb85
    quantum_defects: ClassVar = {
        (0, 0.5, 1 / 2): [3.1311807, 0.1787, 0, 0, 0],  # [1]
        (1, 0.5, 1 / 2): [2.6548849, 0.29, 0, 0, 0],  # [2]
        (1, 1.5, 1 / 2): [2.6416737, 0.295, 0, 0, 0],  # [2]
        (2, 1.5, 1 / 2): [1.3480948, -0.6054, 0, 0, 0],  # [1]
        (2, 2.5, 1 / 2): [1.3464622, -0.594, 0, 0, 0],  # [1]
        (3, 2.5, 1 / 2): [0.0165192, -0.085, 0, 0, 0],  # [3]
        (3, 3.5, 1 / 2): [0.0165437, -0.086, 0, 0, 0],  # [3]
        (4, 3.5, 1 / 2): [0.004, 0, 0, 0, 0],  # [4]
        (4, 4.5, 1 / 2): [0.004, 0, 0, 0, 0],  # [4]
    }
