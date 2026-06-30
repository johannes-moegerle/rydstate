from typing import ClassVar

from rydstate.species.sqdt import SQDT


class SQDTStrontium88(SQDT):
    species = "Sr88"
    is_default = True
    nist_data_file = "nist_data.txt"

    # https://webbook.nist.gov/cgi/inchi?ID=C7440246&Mask=20
    ionization_energy = (5.694_84, "eV")

    # -- [1] Brienza 2023, Phys. Rev. A 108, 022815
    #        Microwave spectroscopy of low-l singlet strontium Rydberg states at intermediate n
    #        Isotope Sr84
    # -- [2] Patsch 2021, http://dx.doi.org/10.17169/refubium-34581
    #        Dissertation: Control of Rydberg atoms for quantum technologies
    #        see table A.2 (and A.3)
    #        Isotope not specified (probably Sr88)
    # -- [3] Robertson 2021, Comput. Phys. Commun. 45, 107814 (2021)
    #        ARC 3.0: An expanded Python toolbox for atomic physics calculations
    #        see table B.1
    #        Isotope Sr88
    # -- [4] (not used but also lot of data) Vaillant 2012, J. Phys. B: At. Mol. Opt. Phys. 45 135004
    #        Long-range Rydberg-Rydberg interactions in calcium, strontium and ytterbium
    quantum_defects: ClassVar = {
        # singlet
        (0, 0.0, 0): (3.2688559, -0.0879, -3.36, 0.0, 0.0),  # [1]
        (1, 1.0, 0): (2.7314851, -5.1501, -140.0, 0.0, 0.0),  # [1]
        (2, 2.0, 0): (2.3821857, -40.5009, -878.6, 0.0, 0.0),  # [1]
        (3, 3.0, 0): (0.0873868, -1.5446, 7.56, 0.0, 0.0),  # [1]
        (4, 4.0, 0): (0.038, 0.0, 0.0, 0.0, 0.0),  # [2]
        (5, 5.0, 0): (0.0134759, 0.0, 0.0, 0.0, 0.0),  # [2]
        # triplet
        (0, 1.0, 1): (3.370773, 0.420, -0.4, 0.0, 0.0),  # [3]
        (1, 0.0, 1): (2.8867, 0.43, -1.8, 0.0, 0.0),  # [3]
        (1, 1.0, 1): (2.8826, 0.39, -1.1, 0.0, 0.0),  # [3]
        (1, 2.0, 1): (2.882, -2.5, 100, 0.0, 0.0),  # [3]
        (2, 1.0, 1): (2.67524, -13.23, -4420, 0.0, 0.0),  # [3]
        (2, 2.0, 1): (2.66149, -16.9, -6630, 0.0, 0.0),  # [3]
        (2, 3.0, 1): (2.655, -65, -13577, 0.0, 0.0),  # [3]
        (3, 2.0, 1): (0.120, -2.2, 100, 0.0, 0.0),  # [3]
        (3, 3.0, 1): (0.119, -2.0, 100, 0.0, 0.0),  # [3]
        (3, 4.0, 1): (0.120, -2.4, 120, 0.0, 0.0),  # [3]
    }


class SQDTStrontium87(SQDT):
    species = "Sr87"
    is_default = True
