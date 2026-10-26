# SQDT models of Yb174_ion

Each json file describes one `ModelSQDTNist`, i.e. one channel by its NIST energy levels (`../nist_data.txt`,
Yb II levels referenced to the Yb+ ground state 4f14.6s 2S1/2), see `ModelSQDTNist.to_dict`.
Each model is used for the principal quantum numbers n in its `n_range` (both bounds inclusive),
i.e. up to the highest n with nu < 15.
There are no quantum defects, i.e. states of these channels outside the `n_range` (or without a NIST level) are not available.
Channels without any model (l >= 5) have zero quantum defects.
