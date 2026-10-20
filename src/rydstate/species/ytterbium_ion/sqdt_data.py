from abc import ABC
from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT


class _SQDTYtterbiumIon(ModelSetSQDT, ABC):
    is_default = True
    # Yb II (Yb+) energy levels, referenced to the Yb+ ground state (4f14.6s 2S1/2)
    nist_data_file = "nist_data.txt"

    # https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Yb&units=1&at_num_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&e_out=0&unc_out=on&biblio=on
    ionization_energy = (12.179185, "eV")

    quantum_defects: ClassVar = None


class SQDTYtterbium171Ion(_SQDTYtterbiumIon):
    species = "Yb171_ion"


class SQDTYtterbium173Ion(_SQDTYtterbiumIon):
    species = "Yb173_ion"


class SQDTYtterbium174Ion(_SQDTYtterbiumIon):
    species = "Yb174_ion"
