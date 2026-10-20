from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class SQDTLithium(ModelSetSQDT):
    species = "Li"
    is_default = True
    nist_data_file = "nist_data.txt"

    # -- [1] Phys. Rev. A 34, 2889 (1986) (Li 7)
    # -- [2] T. F. Gallagher, ``Rydberg Atoms'', Cambridge University Press (2005), ISBN: 978-0-52-102166-1
    # -- [3] Johansson I 1958 Ark. Fysik 15 169
    models: ClassVar = [
        ModelSQDT("Li", "ns_1/2", [0.3995101, 0.029, 0, 0, 0], reference="Phys. Rev. A 34, 2889 (1986) (Li 7)"),
        ModelSQDT("Li", "np_1/2", [0.0471780, -0.024, 0, 0, 0], reference="Phys. Rev. A 34, 2889 (1986) (Li 7)"),
        ModelSQDT("Li", "np_3/2", [0.0471665, -0.024, 0, 0, 0], reference="Phys. Rev. A 34, 2889 (1986) (Li 7)"),
        ModelSQDT(
            "Li",
            "nd_3/2",
            [0.002129, -0.01491, 0.1759, -0.8507, 0],
            reference=(
                "T. F. Gallagher, Rydberg Atoms, Cambridge University Press (2005), ISBN: 978-0-52-102166-1",
                "Johansson I 1958 Ark. Fysik 15 169",
            ),
        ),
        ModelSQDT(
            "Li",
            "nd_5/2",
            [0.002129, -0.01491, 0.1759, -0.8507, 0],
            reference=(
                "T. F. Gallagher, Rydberg Atoms, Cambridge University Press (2005), ISBN: 978-0-52-102166-1",
                "Johansson I 1958 Ark. Fysik 15 169",
            ),
        ),
        ModelSQDT(
            "Li",
            "nf_5/2",
            [0.000305, -0.00126, 0, 0, 0],
            reference=(
                "T. F. Gallagher, Rydberg Atoms, Cambridge University Press (2005), ISBN: 978-0-52-102166-1",
                "Johansson I 1958 Ark. Fysik 15 169",
            ),
        ),
        ModelSQDT(
            "Li",
            "nf_7/2",
            [0.000305, -0.00126, 0, 0, 0],
            reference=(
                "T. F. Gallagher, Rydberg Atoms, Cambridge University Press (2005), ISBN: 978-0-52-102166-1",
                "Johansson I 1958 Ark. Fysik 15 169",
            ),
        ),
    ]
