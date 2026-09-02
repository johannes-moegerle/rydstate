import logging

import pytest
from rydstate import RydbergStateSQDT
from rydstate.angular import AngularKetLS
from rydstate.species import get_element_properties, get_potential_class, get_sqdt
from rydstate.species.element_properties import ElementProperties
from rydstate.species.utils import calc_energy_from_nu, calc_nu_from_energy, get_all_subclasses

ALL_AVAILABLE_ION_SPECIES = sorted([cls.species for cls in get_all_subclasses(ElementProperties) if cls.net_charge > 1])


@pytest.mark.parametrize("species", ALL_AVAILABLE_ION_SPECIES)
def test_ion_element_properties(species: str) -> None:
    element_properties = get_element_properties(species)
    assert element_properties.net_charge == 2  # electron orbits the doubly-charged Yb2+ core
    assert element_properties.number_valence_electrons == 1  # alkali-like above the closed-shell core
    assert element_properties.s_c == 0.0


def test_energy_nu_helpers_scale_with_charge_squared() -> None:
    # E = -1/2 Z^2 mu / nu^2, and nu is the inverse relation
    assert calc_energy_from_nu(1.0, 50, charge=2) == pytest.approx(-0.5 * 4 / 2500)
    assert calc_nu_from_energy(1.0, calc_energy_from_nu(1.0, 37.0, charge=2), charge=2) == pytest.approx(37.0)
    # charge=1 (default) reproduces the neutral-atom relation
    assert calc_energy_from_nu(1.0, 50) == pytest.approx(-0.5 / 2500)


@pytest.mark.parametrize("species", ALL_AVAILABLE_ION_SPECIES)
def test_ion_coulomb_potential_uses_net_charge(species: str) -> None:
    potential = get_potential_class(species)(0)
    distance = 998.5
    assert potential.calc_potential_coulomb(distance) == pytest.approx(-2 / distance)


@pytest.mark.parametrize("species", ALL_AVAILABLE_ION_SPECIES)
def test_ion_nist_data_is_loaded(species: str) -> None:
    sqdt = get_sqdt(species)
    levels = sqdt._nist_energy_levels  # noqa: SLF001
    assert len(levels) > 0

    if "Yb" in species:
        # the parser must pick up the single-electron Yb+ levels over the 4f14 core
        assert levels[(6, 0, 0.5, 0.5)] == 0.0  # 4f14.6s 2S1/2 ground state
        assert (6, 1, 0.5, 0.5) in levels  # 4f14.6p 2P1/2
        assert (5, 2, 1.5, 0.5) in levels  # 4f14.5d 2D3/2
        # only doublets (s_tot = 1/2) since Yb+ is alkali-like (closed-shell 4f14 core)
        assert all(s_tot == 0.5 for (_n, _l, _j, s_tot) in levels)


# Low-lying Sr+ states (n, l_r) whose energies come from the NIST data (nu is taken from the measured levels
# and the radial shape from the model potential).
# The expected number of radial nodes of a state is n - l - 1.
STRONTIUM_ION_LOW_LYING_STATES = [
    (5, 0),  # 5s, ground state of Sr+
    (6, 0),  # 6s
    (7, 0),  # 7s
    (5, 1),  # 5p
    (6, 1),  # 6p
    (4, 2),  # 4d
    (5, 2),  # 5d
    (4, 3),  # 4f
]


@pytest.mark.parametrize(("n", "l_r"), STRONTIUM_ION_LOW_LYING_STATES)
def test_strontium_ion_radial_wavefunction(n: int, l_r: int, caplog: pytest.LogCaptureFixture) -> None:
    """The integrated Sr+ wavefunction is normalized, has n - l - 1 nodes and raises no warnings."""
    j_tot = l_r + 0.5
    angular = AngularKetLS(l_r=l_r, j_tot=j_tot, f_tot=j_tot, species="Sr88_ion")
    state = RydbergStateSQDT("Sr88_ion", n=n, angular=angular)
    radial = state.radial

    with caplog.at_level(logging.WARNING):
        radial.integrate_wavefunction()

    # No sanity-check (or any other) warning should be emitted during the integration.
    warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert warnings == [], f"Unexpected warnings for Sr88_ion n={n} l={l_r}:\n" + "\n".join(warnings)

    # The wavefunction is correctly integrated and normalized ...
    assert radial.norm == pytest.approx(1.0)
    # ... and has exactly n - l - 1 nodes with the (sign-corrected) model potential.
    assert radial.nodes == n - l_r - 1


# Measured energies (in 1/cm above the Sr+ ground state) of 88Sr+ nP1/2 Rydberg states of a single trapped ion,
# F. Pokorny, Doctoral thesis, Stockholm University (2020), Table 5.1 (experimental uncertainty 0.0016 1/cm).
STRONTIUM_ION_P_ENERGIES_POKORNY_2020 = {
    48: 88_754.5995,
    50: 88_771.8845,
    52: 88_787.1237,
    56: 88_812.6512,
    57: 88_818.1741,
}


@pytest.mark.parametrize(("n", "energy_cm"), STRONTIUM_ION_P_ENERGIES_POKORNY_2020.items())
def test_strontium_ion_p_state_energies(n: int, energy_cm: float) -> None:
    """The Sr88_ion quantum defects reproduce the measured nP1/2 Rydberg energies (~ 50 MHz level)."""
    state = RydbergStateSQDT("Sr88_ion", n=n, l_r=1, j_tot=0.5, f_tot=0.5)
    assert state.get_energy("1/cm") == pytest.approx(energy_cm, abs=0.005)


@pytest.mark.parametrize(("n", "quantum_defect"), [(24, 1.4563), (27, 1.4563)])
def test_strontium_ion_d_state_quantum_defects(n: int, quantum_defect: float) -> None:
    """The Sr88_ion nD3/2 quantum defects agree with the trapped-ion measurements of G. Higgins (thesis, 2018)."""
    state = RydbergStateSQDT("Sr88_ion", n=n, l_r=2, j_tot=1.5, f_tot=1.5)
    assert n - state.nu == pytest.approx(quantum_defect, abs=0.001)


def test_strontium_ion_s_state_quantum_defect() -> None:
    """The Sr88_ion nS1/2 quantum defect at high n matches the measured value mu(I++) = 2.7062(2) (Pokorny 2020)."""
    state = RydbergStateSQDT("Sr88_ion", n=60, l_r=0, j_tot=0.5, f_tot=0.5)
    assert 60 - state.nu == pytest.approx(2.7062, abs=0.0005)


def test_strontium_ion_nist_levels_match_quantum_defects() -> None:
    """For the highest NIST levels (n >= 9), nu from the NIST energies and from the quantum defects agree."""
    sqdt = get_sqdt("Sr88_ion")
    levels = sqdt._nist_energy_levels  # noqa: SLF001
    for (n, l_r, j_tot, s_tot), energy in levels.items():
        if n < 9:
            continue
        if (n, l_r) == (10, 2):
            # The NIST 10d fine structure is inverted (J=5/2 below J=3/2, unlike all other nd levels),
            # so the 10d levels deviate from the otherwise smooth nd series by ~0.006 in nu.
            continue
        if j_tot == l_r + 0.5 and levels.get((n, l_r, l_r - 0.5, s_tot)) == energy:
            # Fine structure not resolved in NIST (identical energies for both J): only compare the J = l - 1/2 entry
            continue
        state = RydbergStateSQDT("Sr88_ion", n=n, l_r=l_r, j_tot=j_tot, f_tot=j_tot)
        nu_nist = sqdt.calc_nu(n, state.angular, use_nist_data=True)
        nu_qd = sqdt.calc_nu(n, state.angular, use_nist_data=False)
        assert nu_nist == pytest.approx(nu_qd, abs=0.007), f"n={n}, l={l_r}, j={j_tot}"
