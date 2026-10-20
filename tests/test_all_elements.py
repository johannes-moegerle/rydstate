from typing import Any

import pytest
from rydstate import RydbergStateSQDT, RydbergStateSQDTDivalent
from rydstate.angular import AngularKetLS
from rydstate.element_properties import get_element_properties
from rydstate.qdt import get_sqdt
from rydstate.qdt.sqdt import ModelSetSQDT
from rydstate.utils import get_all_subclasses

ALL_AVAILABLE_SQDT_SPECIES = sorted([cls.species for cls in get_all_subclasses(ModelSetSQDT)])


@pytest.mark.parametrize("species", ALL_AVAILABLE_SQDT_SPECIES)
def test_sqdt_species(species: str) -> None:
    element_properties = get_element_properties(species)
    sqdt = get_sqdt(species)
    if sqdt.models is None:
        pytest.skip(f"Species {species} does not have quantum defects defined.")
    i_c = element_properties.i_c

    state: RydbergStateSQDT[Any]
    if element_properties.number_valence_electrons == 1:
        angular = AngularKetLS(l_r=0, f_tot=i_c + 0.5, species=species)
        state = RydbergStateSQDT(species, n=50, angular=angular)
        state.radial.integrate_wavefunction()
        with pytest.raises(ValueError, match="Unknown quantum numbers detected"):
            AngularKetLS(l_r=1, species=species)
    elif element_properties.number_valence_electrons == 2:
        for s_tot in [0, 1]:
            state = RydbergStateSQDTDivalent(species, n=50, l=1, s=s_tot, j=1 + s_tot, f=s_tot + 1 + i_c)
            state.radial.integrate_wavefunction()


def test_sqdt_total_energy_does_not_require_nu_below_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """A state energy remains defined even when its reference-specific nu is not."""
    species = "Rb"
    angular = AngularKetLS(l_r=0, f_tot=0.5, species=species)
    state = RydbergStateSQDT(species, n=60, angular=angular)
    # note: the model is shared by all states of the channel, so we have to clean up carefully (monkeypatch restores it)
    model = state.model
    threshold_au = model.reference_ionization_threshold_au
    with monkeypatch.context() as patch:
        patch.setattr(model, "reference_ionization_threshold_au", threshold_au - 1e-3)

        binding_energy_au = state.get_binding_energy("a.u.")
        assert binding_energy_au < 0
        assert state.get_energy("a.u.") == pytest.approx(threshold_au + binding_energy_au)
        with pytest.raises(ValueError, match="nu is not defined"):
            _ = state.nu
    assert model.reference_ionization_threshold_au == threshold_au
