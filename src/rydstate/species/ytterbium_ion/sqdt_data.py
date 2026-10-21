from abc import ABC

from rydstate.qdt.sqdt import ModelSetSQDT


class _SQDTYtterbiumIon(ModelSetSQDT, ABC):
    # Yb II (Yb+) energy levels, referenced to the Yb+ ground state (4f14.6s 2S1/2)
    nist_data_file = "nist_data.txt"


class SQDTYtterbium171Ion(_SQDTYtterbiumIon):
    species = "Yb171_ion"


class SQDTYtterbium173Ion(_SQDTYtterbiumIon):
    species = "Yb173_ion"


class SQDTYtterbium174Ion(_SQDTYtterbiumIon):
    species = "Yb174_ion"
