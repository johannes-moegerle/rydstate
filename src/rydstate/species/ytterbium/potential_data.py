from rydstate.radial.potential import PotentialCoulomb, PotentialFei2009


class PotentialCoulombYtterbium171(PotentialCoulomb):
    species = "Yb171"


class PotentialCoulombYtterbium173(PotentialCoulomb):
    species = "Yb173"


class PotentialCoulombYtterbium174(PotentialCoulomb):
    species = "Yb174"


class _PotentialFei2009YtterbiumAbstract(PotentialFei2009):
    is_default = True

    model_potential_parameter_fei_2009 = (0.8704, 22.0040, 0.1513, 0.3306)


class PotentialFei2009Ytterbium171(_PotentialFei2009YtterbiumAbstract):
    species = "Yb171"


class PotentialFei2009Ytterbium173(_PotentialFei2009YtterbiumAbstract):
    species = "Yb173"


class PotentialFei2009Ytterbium174(_PotentialFei2009YtterbiumAbstract):
    species = "Yb174"
