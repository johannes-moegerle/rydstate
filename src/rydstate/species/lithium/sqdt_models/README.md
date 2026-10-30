# SQDT models of Li

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive).

The ns and np models are fitted to measured levels and intervals of 7Li with `rydstate.fit`, see `fits/lithium/` in
the repository (rerun with `fit_sqdt fits/lithium/fit_config.json`); each file lists the publications of its data as
reference and how it was fitted (`fit_info`), the covariance of the quantum defects is in the fit report
`fits/lithium/fit_report.json` (unscaled, i.e. multiply it by the `reduced_chi2` of the report). They are used from
n = 7 on: the measured levels start at n = 18 (p) and 22 (s), but the NIST levels above n = 6 are far less accurate
than the extrapolated models (see the fit config).

The nd and nf models are not fitted (the available measured d and f levels determine their quantum defects less
precisely than these values); they are used from n = 13 (nd) and 6 (nf), with the NIST levels below.

## References of the nd and nf models

1. T. F. Gallagher, Rydberg Atoms, Cambridge University Press (2005), ISBN: 978-0-52-102166-1
2. Johansson I 1958 Ark. Fysik 15 169
