# SQDT models of Na

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`) for nu < 15,
the other files by their quantum defects (for nu >= 15, if the channel also has NIST levels).

## References

1. Phys. Rev. A 45, 4720 (1992)
2. Quantum Electron. 25 914 (1995)
3. J. Phys. B: At. Mol. Opt. Phys. 30 2345 (1997)
