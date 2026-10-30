# SQDT models of K

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the quantum defects from n = 10 for ns, 9 for np, 6 for nd (the lowest n from which the fit describes the measured
levels), from the next n after the NIST levels for nf (15) and ng (6), and the NIST levels below.
The f and g fine structure is not resolved by the data: the models of nf_5/2 and ng_7/2 are those of nf_7/2 and
ng_9/2.

The quantum defect models are fitted to measured levels and intervals of 39K with `rydstate.fit`, see
`fits/potassium/` in the repository (rerun with `fit_sqdt fits/potassium/fit_config.json`); each file lists the
publications of its data as reference and how it was fitted (`fit_info`). The covariance of the quantum defects is
in the fit report `fits/potassium/fit_report.json`; it is unscaled, i.e. multiply it by the `reduced_chi2` of the
report.
