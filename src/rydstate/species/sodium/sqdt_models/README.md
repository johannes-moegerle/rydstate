# SQDT models of Na

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the quantum defects from n = 11 for ns, np, ng, nh and from n = 10 for nd, nf, and the NIST levels below.
The measured levels start at n = 21-23; below them the quantum defects are used down to the lowest n at which they
agree with independent levels (see `fits/sodium/fit_config.json`).

The quantum defect models are fitted to measured microwave intervals between Rydberg levels with `rydstate.fit`,
see `fits/sodium/` in the repository (rerun with `fit_sqdt fits/sodium/fit_config.json`); each file lists the
publications of its data as reference and how it was fitted (`fit_info`). The covariance of the quantum defects is
in the fit report `fits/sodium/fit_report.json`; it is unscaled, i.e. multiply it by the `reduced_chi2` of the report.
