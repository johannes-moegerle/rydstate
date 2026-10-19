from rydstate.radial.potential import PotentialCoulomb, PotentialFei2009


class PotentialCoulombStrontium87(PotentialCoulomb):
    species = "Sr87"


class PotentialCoulombStrontium88(PotentialCoulomb):
    species = "Sr88"


class _PotentialFei2009StrontiumAbstract(PotentialFei2009):
    is_default = True

    model_potential_parameter_fei_2009 = (0.9959, 16.9567, 0.2648, 0.1439)


class PotentialFei2009Strontium87(_PotentialFei2009StrontiumAbstract):
    species = "Sr87"


class PotentialFei2009Strontium88(_PotentialFei2009StrontiumAbstract):
    species = "Sr88"
