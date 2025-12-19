from typing import TYPE_CHECKING

import pytest
from rydstate import RydbergStateSQDTAlkali, RydbergStateSQDTAlkalineLS
from rydstate.species import SpeciesObject

if TYPE_CHECKING:
    from rydstate.rydberg.rydberg_sqdt import RydbergStateSQDTBase


@pytest.mark.parametrize("species_name", SpeciesObject.get_available_species())
def test_magnetic(species_name: str) -> None:
    """Test magnetic units."""
    species = SpeciesObject.from_name(species_name)

    state: RydbergStateSQDTBase
    if species.number_valence_electrons == 1:
        if species.i_c is None:
            state = RydbergStateSQDTAlkali(species, n=50, l=0)
        else:
            state = RydbergStateSQDTAlkali(species, n=50, l=0, f=species.i_c + 0.5)
        state.radial.create_wavefunction()
        with pytest.raises(ValueError, match="j must be set"):
            RydbergStateSQDTAlkali(species, n=50, l=1)
    elif species.number_valence_electrons == 2 and species._quantum_defects is not None:  # noqa: SLF001
        for s_tot in [0, 1]:
            state = RydbergStateSQDTAlkalineLS(species, n=50, l=1, s_tot=s_tot, j_tot=1 + s_tot)
            state.radial.create_wavefunction()
