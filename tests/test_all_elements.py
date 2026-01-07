from typing import TYPE_CHECKING

import pytest
from rydstate import RydbergStateSQDTAlkali, RydbergStateSQDTAlkalineLS
from rydstate.species import SpeciesObject

if TYPE_CHECKING:
    from rydstate import RydbergStateSQDT


@pytest.mark.parametrize("species_name", SpeciesObject.get_available_species())
def test_magnetic(species_name: str) -> None:
    """Test magnetic units."""
    species = SpeciesObject.from_name(species_name)
    i_c = species.i_c if species.i_c is not None else 0

    state: RydbergStateSQDT
    if species.number_valence_electrons == 1:
        state = RydbergStateSQDTAlkali(species, n=50, l=0, f=i_c + 0.5)
        state.radial.create_wavefunction()
        with pytest.raises(ValueError, match="Invalid combination of angular quantum numbers provided"):
            RydbergStateSQDTAlkali(species, n=50, l=1)
    elif species.number_valence_electrons == 2 and species._quantum_defects is not None:  # noqa: SLF001
        for s_tot in [0, 1]:
            state = RydbergStateSQDTAlkalineLS(species, n=50, l=1, s_tot=s_tot, j_tot=1 + s_tot, f_tot=s_tot + 1 + i_c)
            state.radial.create_wavefunction()
