# SQDT models of Cs

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`) for nu < 15,
the other files by their quantum defects (for nu >= 15, if the channel also has NIST levels).

## References

1. Phys. Rev. A 93, 013424 (2016)
2. Phys. Rev. A 26, 2733 (1982)
3. Phys. Rev. A 35, 4650 (1987)
