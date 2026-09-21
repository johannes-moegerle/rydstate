from typing import ClassVar

from rydstate.species.sqdt import SQDT
from rydstate.units import rydberg_constant_au


class SQDTHydrogen(SQDT):
    species = "H"
    is_default = True

    # https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=H&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
    ionization_energy = (13.598_434_599_702, "eV")

    quantum_defects: ClassVar = {}


class SQDTHydrogenTextBook(SQDT):
    species = "H_textbook"
    is_default = True

    ionization_energy = (rydberg_constant_au, "a.u.")

    quantum_defects: ClassVar = {}
