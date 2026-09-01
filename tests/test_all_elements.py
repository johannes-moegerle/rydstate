from typing import Any

import pytest
from rydstate import RydbergStateSQDT
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
            angular = AngularKetLS(l_r=1, s_tot=s_tot, j_tot=1 + s_tot, f_tot=s_tot + 1 + i_c, species=species)
            state = RydbergStateSQDT(species, n=50, angular=angular)
            state.radial.integrate_wavefunction()
