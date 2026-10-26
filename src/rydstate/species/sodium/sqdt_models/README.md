# SQDT models of Na

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the NIST levels up to the highest n with nu < 15, and the quantum defects from the next n on.

## References

1. Phys. Rev. A 45, 4720 (1992)
2. Quantum Electron. 25 914 (1995)
3. J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
