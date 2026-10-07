from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTStrontium88(SQDT):
    species = "Sr88"
    is_default = True
    nist_data_file = "nist_data.txt"

    # Couturier 2019, Phys. Rev. A 99, 022503 (https://doi.org/10.1103/PhysRevA.99.022503)
    # I = 1_377_012_721(10) MHz (= 45932.2002 1/cm, same value as used for MQDTStrontium88)
    ionization_energy = (1_377_012_721, "MHz")

    # -- [1] Brienza 2023, Phys. Rev. A 108, 022815
    #        Microwave spectroscopy of low-l singlet strontium Rydberg states at intermediate n
    #        Isotope Sr84
    # -- [2] Patsch 2021, http://dx.doi.org/10.17169/refubium-34581
    #        Dissertation: Control of Rydberg atoms for quantum technologies
    #        see table A.2 (and A.3)
    #        Isotope not specified (probably Sr88)
    # -- [3] Robertson 2021, Comput. Phys. Commun. 261, 107814 (2021)
    #        ARC 3.0: An expanded Python toolbox for atomic physics calculations
    #        we use the unrounded values of the ARC source code (arc/divalent_atom_data.py,
    #        Strontium88.quantumDefect) instead of the rounded ones of table B.1,
    #        e.g. the rounded 3F2 values deviate by up to 0.2 1/cm from the NIST levels
    #        Isotope Sr88
    # -- [4] (not used but also lot of data) Vaillant 2012, J. Phys. B: At. Mol. Opt. Phys. 45 135004
    #        Long-range Rydberg-Rydberg interactions in calcium, strontium and ytterbium
    quantum_defects: ClassVar = {
        # singlet
        (0, 0.0, 0): [3.2688559, -0.0879, -3.36, 0.0, 0.0],  # [1]
        (1, 1.0, 0): [2.7314851, -5.1501, -140.0, 0.0, 0.0],  # [1]
        (2, 2.0, 0): [2.3821857, -40.5009, -878.6, 0.0, 0.0],  # [1]
        (3, 3.0, 0): [0.0873868, -1.5446, 7.56, 0.0, 0.0],  # [1]
        (4, 4.0, 0): [0.038, 0.0, 0.0, 0.0, 0.0],  # [2]
        (5, 5.0, 0): [0.0134759, 0.0, 0.0, 0.0, 0.0],  # [2]
        # triplet
        (0, 1.0, 1): [3.3707725, 0.41979, -0.421377, 0.0, 0.0],  # [3]
        (1, 0.0, 1): [2.88673, 0.433745, -1.800, 0.0, 0.0],  # [3]
        (1, 1.0, 1): [2.88265, 0.39398, -1.1199, 0.0, 0.0],  # [3]
        (1, 2.0, 1): [2.88163, -2.462, 145.18, 0.0, 0.0],  # [3]
        (2, 1.0, 1): [2.675236, -13.23217, -4418.0, 0.0, 0.0],  # [3]
        (2, 2.0, 1): [2.661488, -16.8524, -6629.26, 0.0, 0.0],  # [3]
        (2, 3.0, 1): [2.655, -65.317, -13576.7, 0.0, 0.0],  # [3]
        (3, 2.0, 1): [0.120588, -2.1847, 102.98, 0.0, 0.0],  # [3]
        (3, 3.0, 1): [0.11899, -2.0446, 103.26, 0.0, 0.0],  # [3]
        (3, 4.0, 1): [0.12000, -2.37716, 118.97, 0.0, 0.0],  # [3]
    }
