from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class SQDTSodium(ModelSetSQDT):
    species = "Na"
    is_default = True
    nist_data_file = "nist_data.txt"

    # Phys. Rev. A 45, 4720 (1992) (https://doi.org/10.1103/PhysRevA.45.4720)
    # Precision Stark spectroscopy of sodium: improved values for the ionization limit and bound states
    # I = 41449.451(2) 1/cm (the same reference as [1] below, which the quantum defects are taken from)
    ionization_energy = (41449.451, "1/cm")

    # -- [1] Phys. Rev. A 45, 4720 (1992)
    # -- [2] Quantum Electron. 25 914 (1995)
    # -- [3] J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
    models: ClassVar = [
        ModelSQDT(
            "Na",
            "ns_1/2",
            [1.34796938, 0.0609892, 0.0196743, -0.001045, 0],
            reference=("Phys. Rev. A 45, 4720 (1992)", "Quantum Electron. 25 914 (1995)"),
        ),
        ModelSQDT(
            "Na", "np_1/2", [0.85544502, 0.112067, 0.0479, 0.0457, 0], reference="Quantum Electron. 25 914 (1995)"
        ),
        ModelSQDT(
            "Na", "np_3/2", [0.85462615, 0.112344, 0.0497, 0.0406, 0], reference="Quantum Electron. 25 914 (1995)"
        ),
        ModelSQDT(
            "Na",
            "nd_3/2",
            [0.014909286, -0.042506, 0.00840, 0, 0],
            reference=("Quantum Electron. 25 914 (1995)", "J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)"),
        ),
        ModelSQDT(
            "Na",
            "nd_5/2",
            [0.01492422, -0.042585, 0.00840, 0, 0],
            reference=("Quantum Electron. 25 914 (1995)", "J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)"),
        ),
        ModelSQDT(
            "Na",
            "nf_5/2",
            [0.001632977, -0.0069906, 0.00423, 0, 0],
            reference="J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)",
        ),
        ModelSQDT(
            "Na",
            "nf_7/2",
            [0.001630875, -0.0069824, 0.00352, 0, 0],
            reference="J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)",
        ),
        ModelSQDT(
            "Na", "ng_7/2", [0.00043825, -0.00283, 0, 0, 0], reference="J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)"
        ),
        ModelSQDT(
            "Na", "ng_9/2", [0.00043740, -0.00297, 0, 0, 0], reference="J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)"
        ),
        ModelSQDT(
            "Na", "nh_9/2", [0.00016114, -0.00185, 0, 0, 0], reference="J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)"
        ),
        ModelSQDT(
            "Na", "nh_5.5", [0.00015796, -0.00148, 0, 0, 0], reference="J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)"
        ),
    ]
