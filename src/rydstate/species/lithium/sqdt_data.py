from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTLithium(SQDT):
    species = "Li"
    is_default = True
    nist_data_file = "nist_data.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7439932&Mask=20
    ionization_energy = (5.391_72, "eV")

    # -- [1] Phys. Rev. A 34, 2889 (1986) (Li 7)
    # -- [2] T. F. Gallagher, ``Rydberg Atoms'', Cambridge University Press (2005), ISBN: 978-0-52-102166-1
    # -- [3] Johansson I 1958 Ark. Fysik 15 169
    quantum_defects: ClassVar = {
        (0, 0.5, 1 / 2): [0.3995101, 0.029, 0, 0, 0],  # [1]
        (1, 0.5, 1 / 2): [0.0471780, -0.024, 0, 0, 0],  # [1]
        (1, 1.5, 1 / 2): [0.0471665, -0.024, 0, 0, 0],  # [1]
        (2, 1.5, 1 / 2): [0.002129, -0.01491, 0.1759, -0.8507, 0],  # [2,3]
        (2, 2.5, 1 / 2): [0.002129, -0.01491, 0.1759, -0.8507, 0],  # [2,3]
        (3, 2.5, 1 / 2): [0.000305, -0.00126, 0, 0, 0],  # [2,3]
        (3, 3.5, 1 / 2): [0.000305, -0.00126, 0, 0, 0],  # [2,3]
    }
