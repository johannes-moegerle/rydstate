# SQDT models of Sr88

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the NIST levels up to the highest n with nu < 15, and the quantum defects from the next n on.

## References

1. Brienza 2023, Phys. Rev. A 108, 022815 -
   Microwave spectroscopy of low-l singlet strontium Rydberg states at intermediate n.
   Isotope Sr84.
2. Patsch 2021, http://dx.doi.org/10.17169/refubium-34581 -
   Dissertation: Control of Rydberg atoms for quantum technologies, see table A.2 (and A.3).
   Isotope not specified (probably Sr88).
3. Robertson 2021, Comput. Phys. Commun. 261, 107814 (2021) -
   ARC 3.0: An expanded Python toolbox for atomic physics calculations.
   We use the unrounded values of the ARC source code (arc/divalent_atom_data.py, Strontium88.quantumDefect)
   instead of the rounded ones of table B.1, e.g. the rounded 3F2 values deviate by up to 0.2 1/cm from the NIST levels.
   Isotope Sr88.
4. (Not used, but also lots of data) Vaillant 2012, J. Phys. B: At. Mol. Opt. Phys. 45 135004 -
   Long-range Rydberg-Rydberg interactions in calcium, strontium and ytterbium.
