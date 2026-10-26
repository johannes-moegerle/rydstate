# SQDT models of Li

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the NIST levels up to the highest n with nu < 15, and the quantum defects from the next n on.

## References

1. Phys. Rev. A 34, 2889 (1986) (Li 7)
2. T. F. Gallagher, Rydberg Atoms, Cambridge University Press (2005), ISBN: 978-0-52-102166-1
3. Johansson I 1958 Ark. Fysik 15 169
