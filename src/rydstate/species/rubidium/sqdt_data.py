from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class SQDTRubidium(ModelSetSQDT):
    species = "Rb"
    nist_data_file = "nist_data.txt"

    # -- [1] Phys. Rev. A 83, 052515 (2011) - Rb87
    # -- [2] Phys. Rev. A 67, 052502 (2003) - Rb85
    # -- [3] Phys. Rev. A 74, 054502 (2006) - Rb85
    # -- [4] Phys. Rev. A 74, 062712 (2006) - Rb85
    models: ClassVar = [
        ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787, 0, 0, 0], reference="Phys. Rev. A 83, 052515 (2011) - Rb87"),
        ModelSQDT("Rb", "np_1/2", [2.6548849, 0.29, 0, 0, 0], reference="Phys. Rev. A 67, 052502 (2003) - Rb85"),
        ModelSQDT("Rb", "np_3/2", [2.6416737, 0.295, 0, 0, 0], reference="Phys. Rev. A 67, 052502 (2003) - Rb85"),
        ModelSQDT("Rb", "nd_3/2", [1.3480948, -0.6054, 0, 0, 0], reference="Phys. Rev. A 83, 052515 (2011) - Rb87"),
        ModelSQDT("Rb", "nd_5/2", [1.3464622, -0.594, 0, 0, 0], reference="Phys. Rev. A 83, 052515 (2011) - Rb87"),
        ModelSQDT("Rb", "nf_5/2", [0.0165192, -0.085, 0, 0, 0], reference="Phys. Rev. A 74, 054502 (2006) - Rb85"),
        ModelSQDT("Rb", "nf_7/2", [0.0165437, -0.086, 0, 0, 0], reference="Phys. Rev. A 74, 054502 (2006) - Rb85"),
        ModelSQDT("Rb", "ng_7/2", [0.004, 0, 0, 0, 0], reference="Phys. Rev. A 74, 062712 (2006) - Rb85"),
        ModelSQDT("Rb", "ng_9/2", [0.004, 0, 0, 0, 0], reference="Phys. Rev. A 74, 062712 (2006) - Rb85"),
    ]
