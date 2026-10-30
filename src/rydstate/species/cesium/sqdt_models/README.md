# SQDT models of Cs

Each json file describes one model of one channel, see `ModelSQDT.to_dict` and `ModelSQDTNist.to_dict`:
the `*_nist.json` files describe the channels by their NIST energy levels (`../nist_data.txt`),
the other files by their quantum defects.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive):
the quantum defects from the lowest measured level of the fit on (n = 11 for ns, np1/2, nd, nf5/2, ng7/2,
18 for np3/2, 14 for nf7/2; 12 for ng9/2, whose NIST levels end at 11), and the NIST levels below.

The quantum defect models are fitted to measured levels and intervals with `rydstate.fit`, see `fits/cesium/` in
the repository (rerun with `fit_sqdt fits/cesium/fit_config.json`); each file lists the publications of its
data as reference and how it was fitted (`fit_info`). The covariance of the quantum defects is in the fit report
`fits/cesium/fit_report.json`.
