from rydstate.radial.potential import PotentialCoulomb


class PotentialCoulombHydrogen(PotentialCoulomb):
    species = "H"
    is_default = True


class PotentialCoulombHydrogenTextBook(PotentialCoulomb):
    species = "H_textbook"
    is_default = True
