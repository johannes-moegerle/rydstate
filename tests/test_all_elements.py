from typing import Any

import pytest
from rydstate import RydbergStateSQDT, RydbergStateSQDTDivalent
from rydstate.angular import AngularKetLS
from rydstate.species import get_all_subclasses, get_element_properties, get_sqdt
from rydstate.species.sqdt import SQDT

ALL_AVAILABLE_SQDT_SPECIES = sorted([cls.species for cls in get_all_subclasses(SQDT)])


@pytest.mark.parametrize("species", ALL_AVAILABLE_SQDT_SPECIES)
def test_sqdt_species(species: str) -> None:
    element_properties = get_element_properties(species)
    sqdt = get_sqdt(species)
    if sqdt.quantum_defects is None:
        pytest.skip(f"Species {species} does not have quantum defects defined.")
    i_c = element_properties.i_c

    state: RydbergStateSQDT[Any]
    if element_properties.number_valence_electrons == 1:
        angular = AngularKetLS(l_r=0, f_tot=i_c + 0.5, species=species)
        state = RydbergStateSQDT(species, n=50, angular=angular)
        state.radial.integrate_wavefunction()
        with pytest.raises(ValueError, match="Unknown quantum numbers detected"):
            AngularKetLS(l_r=1, species=species)
    elif element_properties.number_valence_electrons == 2 and sqdt.quantum_defects is not None:
        for s_tot in [0, 1]:
            state = RydbergStateSQDTDivalent(species, n=50, l=1, s=s_tot, j=1 + s_tot, f=s_tot + 1 + i_c)
            state.radial.integrate_wavefunction()


def test_sqdt_total_energy_does_not_require_nu_below_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A state energy remains defined even when its reference-specific nu is not."""
    species = "Rb"
    sqdt = get_sqdt(species)  # note: this is a process wide cached instance, so we have to clean up carefully
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sqdt, "_reference_ionization_energy", (sqdt.ionization_energy_au - 1e-3, "hartree"))
            sqdt.__dict__.pop("reference_ionization_energy_au", None)  # clear the cached_property
            angular = AngularKetLS(l_r=0, f_tot=0.5, species=species)
            state = RydbergStateSQDT(species, n=60, angular=angular, sqdt=sqdt)

            binding_energy_au = state.get_binding_energy("a.u.")
            assert binding_energy_au < 0
            assert state.get_energy("a.u.") == pytest.approx(sqdt.ionization_energy_au + binding_energy_au)
            with pytest.raises(ValueError, match="nu is not defined"):
                _ = state.nu
    finally:
        sqdt.__dict__.pop("reference_ionization_energy_au", None)  # clear the patched cached_property
