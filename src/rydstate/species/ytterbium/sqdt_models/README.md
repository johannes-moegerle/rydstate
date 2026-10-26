# SQDT models of Yb174

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the NIST levels up to the highest n with nu < 15, and the quantum defects from the next n on.
An exception is the 6sns 3S1 channel: its Rydberg-Ritz formula diverges at low n (e.g. nu = 43.2 for n = 14),
so its quantum defects are only used from n = 20 on, where |delta(n) - delta_0| < 1/2,
i.e. the states with n = 14 to 19 between the highest NIST level (6s13s) and n = 20 are not available.
The comments of the models name the table of the reference, from which the quantum defects are taken.

## References

1. Peper 2024, http://arxiv.org/abs/2406.01482 -
   Spectroscopy and modeling of 171Yb Rydberg states for high-fidelity two-qubit gates, see Table I.
   Isotope Yb174.
2. Wilson 2019, https://arxiv.org/abs/1912.08754 -
   Trapped arrays of alkaline earth Rydberg atoms in optical tweezers.
   The fit is only valid from 28s upward, see Table S2, but adjusted for non-Rydberg Ritz.
   Isotope Yb174.
3. Kuroda 2025, https://arxiv.org/abs/2507.11487 -
   Microwave spectroscopy and multi-channel quantum defect analysis of ytterbium Rydberg states, see Table S1 - S8.
   Isotope Yb174.

## Not included

The 6snf +/- states of Table S8 of [3] (which are not LS coupled) are not included:

- (l=4, j=4, "+"): [0.0262659964, 0.0254568575]
- (l=4, j=4, "-"): [-0.148808463, -0.134219071]
