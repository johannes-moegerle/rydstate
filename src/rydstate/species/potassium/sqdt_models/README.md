# SQDT models of K

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`) for nu < 15,
the other files by their quantum defects (for nu >= 15, if the channel also has NIST levels).

## References

1. Phys. Scr. 27, 300 (1983)
2. Opt. Commun. 39, 370 (1981)
3. Ark. Fys., 10 p.583 (1956)
