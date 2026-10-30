# Fits of the SQDT models

The quantum defect models of the species (`src/rydstate/species/<species>/sqdt_models/*.json`) are fitted to
measured energies and intervals of Rydberg levels with `rydstate.fit`. Every fit has its own directory
`<species>/` (named like `src/rydstate/species/<species>`) with one config file, which lists the datasets and all
choices of the fit (selected values, added systematic uncertainties, excluded values, free dataset thresholds,
number of coefficients and the n range of every model), each with its reason:

- `<species>/fit_config.json`: fit config, see `rydstate.fit.fit_config` for its keys.
- `<species>/data/<dataset>/`: one directory per source, see `rydstate.fit.load_measurements`:
  - `measurements.csv`: one measured value per row, `value = E(level) - E(lower_level)` (empty `lower_level`: energy
    above the hyperfine centre of gravity of the ground state, the energy zero of rydstate), with its uncertainty
    and unit, the label of the value in the source and notes. Corrections of the values (energy zero, errata) are
    applied; the choices of a fit are not.
  - `metadata.json`: the reference (with doi), the method and uncertainties of the source, how the values were
    converted and caveats. The source files themselves are not stored here, see the doi.
- `<species>/fit_report.json`: fit report (fit statistics, models, dataset offsets, covariance). The residuals are
  not stored: `fit_sqdt` prints the values with a residual above 3 sigma (the uncertainty used by the fit, i.e.
  including the systematic of the config).

Every fitted model contains the publications of its data as `reference` and a `fit_info` (method, rydstate version,
n range of the data and the config), so it can be traced back to this fit. The uncertainties of the quantum defects
are only given by the covariance in the report: the quantum defects of a model are strongly correlated (and some
with those of other models, e.g. through intervals between series), so their standard deviations alone cannot be
propagated, e.g. to an energy.

## Usage

```bash
fit_sqdt fits/cesium/fit_config.json --dry-run  # fit and print the result
fit_sqdt fits/cesium/fit_config.json            # fit and write the models and the report
fit_sqdt fits/cesium/fit_config.json --check    # check that the written models are reproduced
```

`tests/test_fit_config.py` checks that every config reproduces the models of the species, so a change of the
data or the config needs a rerun of the fit.

## Species

- Cs (`cesium/`): global fit of the s, p, d, f, g series (505 values from 7 datasets, chi2/dof = 2.0), every
  dataset of absolute energies with its own ionization threshold. The measured data were collected and converted
  in the rydstate_qdt project (T002).
