from abc import ABC

from rydstate.qdt.sqdt import ModelSetSQDT


class _SQDTStrontiumIon(ModelSetSQDT, ABC):
    # Sr II (Sr+) energy levels, referenced to the Sr+ ground state (4p6.5s 2S1/2)
    nist_data_file = "nist_data.txt"


class SQDTStrontium87Ion(_SQDTStrontiumIon):
    species = "Sr87_ion"
    # no quantum defects (they would have to account for the hyperfine structure), i.e. only the NIST levels are used


class SQDTStrontium88Ion(_SQDTStrontiumIon):
    species = "Sr88_ion"
    # see sqdt_models/README.md for the references and how the quantum defects are derived from them
    models_directory = "sqdt_models"
