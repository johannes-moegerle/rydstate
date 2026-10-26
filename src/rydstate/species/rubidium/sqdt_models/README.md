# SQDT models of Rb

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the NIST levels up to the highest n with nu < 15, and the quantum defects from the next n on.
Exceptions are the ns and nd channels, whose NIST levels from 13s to 38s and from 11d to 36d are missing:
their quantum defects are used from 13s and 11d on, so the NIST levels above the gap are not used.

## References

1. Phys. Rev. A 83, 052515 (2011) - Rb87
2. Phys. Rev. A 67, 052502 (2003) - Rb85
3. Phys. Rev. A 74, 054502 (2006) - Rb85
4. Phys. Rev. A 74, 062712 (2006) - Rb85
