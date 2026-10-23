# SQDT models of Yb174_ion

Each json file describes one `ModelSQDTNist`, i.e. one channel by its NIST energy levels (`../nist_data.txt`,
Yb II levels referenced to the Yb+ ground state 4f14.6s 2S1/2) for nu < 15, see `ModelSQDTNist.to_dict`.
There are no quantum defects, i.e. states of these channels with nu >= 15 (or without a NIST level) are not available.
Channels without any model (l >= 5) have zero quantum defects.
Yb171_ion and Yb173_ion have no SQDT models yet (they would have to account for the hyperfine structure).
