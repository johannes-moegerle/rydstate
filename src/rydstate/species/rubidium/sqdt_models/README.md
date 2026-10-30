# SQDT models of Rb

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the quantum defects from n = 13 for ns and np (the lowest measured np levels; the NIST ns levels from 13s to 38s are
missing), from n = 11 for nd (the NIST nd levels from 11d to 36d are missing), from the next n after the NIST levels
for nf (9) and ng (6), and the NIST levels below.

The quantum defect models are fitted to measured levels and intervals of 87Rb and 85Rb (converted to 87Rb) with
`rydstate.fit`, see `fits/rubidium/` in the repository (rerun with `fit_sqdt fits/rubidium/fit_config.json`); each
file lists the publications of its data as reference and how it was fitted (`fit_info`). The covariance of the
quantum defects is in the fit report `fits/rubidium/fit_report.json`; it is unscaled, i.e. multiply it by the
`reduced_chi2` of the report.
