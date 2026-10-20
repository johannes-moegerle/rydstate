from abc import ABC
from typing import ClassVar

from rydstate.angular.core_ket import CoreKet
from rydstate.element_properties import ElementProperties

# Double ionization threshold I++ (Sr+ -> Sr2+), referenced to the Sr+ ground state, from a Rydberg-Ritz fit
# to the nS1/2 Rydberg series (n ~ 30 - 60) of a single trapped 88Sr+ ion measured with a 2 MHz wavelength
# meter [Pokorny 2020, Doctoral thesis, Stockholm University, ISBN 978-91-7911-293-6].
# Consistent with the earlier trapped-ion value 88 965.022(11) cm^-1 [Higgins 2018, Doctoral thesis, Stockholm
# University, http://urn.kb.se/resolve?urn=urn:nbn:se:su:diva-153415].
# Note: NIST (https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=Sr+II) adopts 88 965.18(2) cm^-1 from the
# pulsed-laser spectroscopy (0.2 cm^-1 absolute accuracy) of Lange et al. [Z. Phys. D 18, 319 (1991)], which is
# 0.16 cm^-1 (~5 GHz) higher. Since the quantum defects of Lange et al. agree with the trapped-ion measurements,
# this is a common offset of their absolute energy scale, which cancels in the quantum defects
# (they only depend on I++ - E).
_IONIZATION_THRESHOLD = (88_965.019_1, "1/cm")


class _ElementPropertiesStrontiumAbstractIon(ElementProperties, ABC):
    Z = 38
    net_charge = 2
    number_valence_electrons = 1
    ground_state_shell = (5, 0)
    additional_allowed_shells: ClassVar = [(4, 2), (4, 3)]
    core_electron_configuration = "4p6"

    # Greene, Aymar Phys. Rev. A 44, 1773, 1991 (https://doi.org/10.1103/PhysRevA.44.1773)
    alpha_closed_shell_core = 7.5
    # fitted to the NIST ASD line strengths of Sr II with accuracy AA / A+ (5s1/2-5p1/2, 5s1/2-5p3/2, 4d3/2-5p1/2),
    # which are reproduced within 1% (the bare dipole operator overestimates them by 7-8%)
    r_c_dipole_operator = 3.46


class ElementPropertiesStrontium87Ion(_ElementPropertiesStrontiumAbstractIon):
    species = "Sr87_ion"
    i_c = 9 / 2

    mass_number = 87
    atomic_mass_u = 86.90887749454

    # https://nds.iaea.org/nuclearmoments/isotope_measurement_results.php?A=87&Z=38
    nuclear_dipole = -1.09316

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=9 / 2, s_c=0, l_c=0): _IONIZATION_THRESHOLD}


class ElementPropertiesStrontium88Ion(_ElementPropertiesStrontiumAbstractIon):
    species = "Sr88_ion"
    i_c = 0

    mass_number = 88
    atomic_mass_u = 87.905612253

    ionization_threshold_dict: ClassVar = {CoreKet(i_c=0, s_c=0, l_c=0): _IONIZATION_THRESHOLD}
