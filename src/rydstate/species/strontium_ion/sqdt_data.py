from abc import ABC
from typing import ClassVar

from rydstate.qdt.sqdt import ModelSetSQDT, ModelSQDT


class _SQDTStrontiumIon(ModelSetSQDT, ABC):
    is_default = True
    # Sr II (Sr+) energy levels, referenced to the Sr+ ground state (4p6.5s 2S1/2)
    nist_data_file = "nist_data.txt"

    # -- [1] Pokorny 2020, Doctoral thesis, Stockholm University, ISBN 978-91-7911-293-6
    #        A microwave dressed Rydberg ion, http://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-184811
    #        Single trapped 88Sr+ ion, I++ = 88 965.0191(16) cm^-1, see section 5.2
    #        nS1/2 series: mu(I++) = 2.7062(2)
    #        nP1/2 states n = 48, 50, 52, 56, 57 (Table 5.1, uncertainty 0.0016 cm^-1); all five tabulated energies
    #        give delta = n - 2 sqrt(R_M / (I++ - E)) = 2.3265(1) (the fit in [1], mu(I++) = 2.31(18), also leaves
    #        the energy dependence free and is therefore much less precise)
    # -- [2] Higgins 2018, Doctoral thesis, Stockholm University, http://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-153415
    #        A single trapped Rydberg ion (also published as Springer Thesis, https://doi.org/10.1007/978-3-030-33770-4)
    #        Single trapped 88Sr+ ion, I++ = 88 965.022(11) cm^-1, see section 4.6 and Table 4.1
    #        nS1/2 series: mu(I++) = 2.7063(9), dmu/dE = -0.04(9) Ryd^-1
    #        mu(24D3/2) = 1.4563(3), mu(27D3/2) = 1.4563(4)
    # -- [3] Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593
    #        Rydberg states of the strontium ion
    #        Sr+ ns (26 <= n <= 79), nd (25 <= n <= 83), nf (28 <= n <= 86), ng (30 <= n <= 81) series
    #        (pulsed lasers, 0.2 cm^-1 accuracy, fine structure not resolved), I++ = 88 965.18(2) cm^-1
    #        Table 2: mu0(I++) and dmu/dE [Ryd^-1]: s: 2.707(2), -0.055(15); d: 1.456(2), -0.037(15);
    #                                                f: 0.069(2), 0.175(15); g: 0.011(2), 0.033(15)
    #        Measured (fine-structure resolved) 7p levels: mu(7p1/2) = 2.348(2), mu(7p3/2) = 2.332(2)
    # -- [4] NIST Atomic Spectra Database, Sr II levels (see nist_data.txt), only used for the (n-independent)
    #        fine-structure splitting of the nd quantum defects: delta(nD3/2) - delta(nD5/2) = 0.0043 (4d - 9d)
    # -- Not used: Djerad 1991, J. Phys. II 1, 1 (https://doi.org/10.1051/jp2:1991135) and
    #        Glukhov 2013, Opt. Spectrosc. 115, 9 (https://doi.org/10.1134/S0030400X13070060) give extended Ritz
    #        parameters fitted to the low-lying tabulated levels only (fine-structure averaged; [3] showed the 7p
    #        levels used by Djerad to be wrong by 34 cm^-1).


class SQDTStrontium87Ion(_SQDTStrontiumIon):
    species = "Sr87_ion"
    # no quantum defects (they would have to account for the hyperfine structure), i.e. only the NIST levels are used


class SQDTStrontium88Ion(_SQDTStrontiumIon):
    species = "Sr88_ion"

    # The quantum defect is expanded as delta(n) = d0 + d2 / (n - d0)^2 + d4 / (n - d0)^4 + ...
    # Reference [3] instead parametrizes the energy dependence linearly around the double ionization threshold,
    # mu(E) = mu0 + dmu/dE (E - I++) with mu0 = mu(I++). Inserting the binding energy of an electron bound to a
    # core of charge Z = 2 (Sr2+), E - I++ = -Z^2 R_M / (n - mu)^2 = -4 / (n - mu)^2 in Rydberg units, gives
    # mu(n) = mu0 - 4 dmu/dE / (n - mu)^2, i.e. d0 = mu0 and d2 = -Z^2 dmu/dE = -4 dmu/dE (dmu/dE in Ryd^-1).
    # (Using (n - mu)^2 instead of (n - d0)^2 only affects d4 and higher, far below the precision of [3].)
    # For the P series no high-n Rydberg-Ritz fit exists in the literature: d0(P1/2) is the high-n value of [1],
    # d0(P3/2) = d0(P1/2) - [mu(7p1/2) - mu(7p3/2)] with the fine-structure splitting from [3], and d2 is chosen
    # such that the measured mu(7p_j) of [3] are reproduced: d2 = [mu(7p_j) - d0] (7 - d0)^2 = 0.47.
    # The fine structure of the nd series is not resolved in [3], so d0(D3/2) is taken from [2] and d0(D5/2)
    # from the fine-structure splitting of the NIST levels [4]; the fine structure of the nf and ng series
    # is neither resolved in [3] nor in the NIST data (n >= 5).
    models: ClassVar = [
        # ns_1/2: d0 [1], d2 [3]
        ModelSQDT(
            "Sr88_ion",
            "ns_1/2",
            [2.7062, 0.22, 0, 0, 0],
            reference=(
                "Pokorny 2020, Doctoral thesis, Stockholm University, ISBN 978-91-7911-293-6",
                "Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
            ),
        ),
        # np_1/2: d0 [1], d2 from mu(7p1/2) [3]
        ModelSQDT(
            "Sr88_ion",
            "np_1/2",
            [2.3265, 0.47, 0, 0, 0],
            reference=(
                "Pokorny 2020, Doctoral thesis, Stockholm University, ISBN 978-91-7911-293-6",
                "Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
            ),
        ),
        # np_3/2: d0 = d0(P1/2) - 0.016 [1, 3], d2 from mu(7p3/2) [3]
        ModelSQDT(
            "Sr88_ion",
            "np_3/2",
            [2.3105, 0.47, 0, 0, 0],
            reference=(
                "Pokorny 2020, Doctoral thesis, Stockholm University, ISBN 978-91-7911-293-6",
                "Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
            ),
        ),
        # nd_3/2: d0 [2], d2 [3]
        ModelSQDT(
            "Sr88_ion",
            "nd_3/2",
            [1.4563, 0.148, 0, 0, 0],
            reference=(
                "Higgins 2018, Doctoral thesis, Stockholm University, http://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-153415",
                "Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
            ),
        ),
        # nd_5/2: d0 = d0(D3/2) - 0.0043 [2, 4], d2 [3]
        ModelSQDT(
            "Sr88_ion",
            "nd_5/2",
            [1.4520, 0.148, 0, 0, 0],
            reference=(
                "Higgins 2018, Doctoral thesis, Stockholm University, http://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-153415",
                "NIST Atomic Spectra Database, Sr II levels (see nist_data.txt)",
                "Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
            ),
        ),
        ModelSQDT(
            "Sr88_ion",
            "nf_5/2",
            [0.069, -0.70, 0, 0, 0],
            reference="Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
        ),
        ModelSQDT(
            "Sr88_ion",
            "nf_7/2",
            [0.069, -0.70, 0, 0, 0],
            reference="Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
        ),
        ModelSQDT(
            "Sr88_ion",
            "ng_7/2",
            [0.011, -0.132, 0, 0, 0],
            reference="Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
        ),
        ModelSQDT(
            "Sr88_ion",
            "ng_9/2",
            [0.011, -0.132, 0, 0, 0],
            reference="Lange 1991, Z. Phys. D 18, 319, https://doi.org/10.1007/BF01426593",
        ),
    ]
