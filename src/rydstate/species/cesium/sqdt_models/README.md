# SQDT models of Cs

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the NIST levels up to the highest n with nu < 15, and the quantum defects from the next n on.

## References

1. Phys. Rev. A 93, 013424 (2016)
2. Phys. Rev. A 26, 2733 (1982)
3. Phys. Rev. A 35, 4650 (1987)
