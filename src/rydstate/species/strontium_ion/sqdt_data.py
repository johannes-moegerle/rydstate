from abc import ABC
from typing import ClassVar

from rydstate.species.sqdt import SQDT


class _SQDTStrontiumIon(SQDT, ABC):
    is_default = True
    # Sr II (Sr+) energy levels, referenced to the Sr+ ground state (4p6.5s 2S1/2)
    nist_data_file = "nist_data.txt"

    # Double ionization threshold I++ (Sr+ -> Sr2+), referenced to the Sr+ ground state, from a Rydberg-Ritz fit
    # to the nS1/2 Rydberg series (n ~ 30 - 60) of a single trapped 88Sr+ ion measured with a 2 MHz wavelength
    # meter [1]. Consistent with the earlier trapped-ion value 88 965.022(11) cm^-1 [2].
    # Note: NIST (https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sr+II) adopts 88 965.18(2) cm^-1 from the
    # pulsed-laser spectroscopy (0.2 cm^-1 absolute accuracy) of Lange et al. [3], which is 0.16 cm^-1 (~5 GHz)
    # higher. Since the quantum defects of [3] agree with [1], [2], this is a common offset of the absolute energy
    # scale of [3], which cancels in the quantum defects (they only depend on I++ - E).
    ionization_energy = (88_965.019_1, "1/cm")

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
    #
    # The quantum defect is expanded as delta(n) = d0 + d2 / (n - d0)^2 + d4 / (n - d0)^4 + ...
    # The linear energy dependence mu(E) = mu0 + dmu/dE (E - I++) of [3] with E - I++ = -Z^2 R_M / (n - mu)^2
    # (Z = 2) corresponds to d0 = mu0 and d2 = -4 dmu/dE (dmu/dE in Ryd^-1).
    # For the P series no high-n Rydberg-Ritz fit exists in the literature: d0(P1/2) is the high-n value of [1],
    # d0(P3/2) = d0(P1/2) - [mu(7p1/2) - mu(7p3/2)] with the fine-structure splitting from [3], and d2 is chosen
    # such that the measured mu(7p_j) of [3] are reproduced: d2 = [mu(7p_j) - d0] (7 - d0)^2 = 0.47.
    # The fine structure of the nd series is not resolved in [3], so d0(D3/2) is taken from [2] and d0(D5/2)
    # from the fine-structure splitting of the NIST levels [4]; the fine structure of the nf and ng series
    # is neither resolved in [3] nor in the NIST data (n >= 5).
    quantum_defects: ClassVar = {
        (0, 0.5, 1 / 2): (2.7062, 0.22, 0, 0, 0),  # d0 [1], d2 [3]
        (1, 0.5, 1 / 2): (2.3265, 0.47, 0, 0, 0),  # d0 [1], d2 from mu(7p1/2) [3]
        (1, 1.5, 1 / 2): (2.3105, 0.47, 0, 0, 0),  # d0 = d0(P1/2) - 0.016 [1, 3], d2 from mu(7p3/2) [3]
        (2, 1.5, 1 / 2): (1.4563, 0.148, 0, 0, 0),  # d0 [2], d2 [3]
        (2, 2.5, 1 / 2): (1.4520, 0.148, 0, 0, 0),  # d0 = d0(D3/2) - 0.0043 [2, 4], d2 [3]
        (3, 2.5, 1 / 2): (0.069, -0.70, 0, 0, 0),  # [3]
        (3, 3.5, 1 / 2): (0.069, -0.70, 0, 0, 0),  # [3]
        (4, 3.5, 1 / 2): (0.011, -0.132, 0, 0, 0),  # [3]
        (4, 4.5, 1 / 2): (0.011, -0.132, 0, 0, 0),  # [3]
    }


class SQDTStrontium87Ion(_SQDTStrontiumIon):
    species = "Sr87_ion"


class SQDTStrontium88Ion(_SQDTStrontiumIon):
    species = "Sr88_ion"
