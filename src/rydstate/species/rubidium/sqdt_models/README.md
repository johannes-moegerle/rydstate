# SQDT models of Rb

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`) for nu < 15,
the other files by their quantum defects (for nu >= 15, if the channel also has NIST levels).

## References

1. Phys. Rev. A 83, 052515 (2011) - Rb87
2. Phys. Rev. A 67, 052502 (2003) - Rb85
3. Phys. Rev. A 74, 054502 (2006) - Rb85
4. Phys. Rev. A 74, 062712 (2006) - Rb85
