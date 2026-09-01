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


# Low-lying Sr+ states (n, l_r) whose energies come from the NIST data (no quantum defects are defined
# for the ion, so nu is taken from the measured levels and the radial shape from the model potential).
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
