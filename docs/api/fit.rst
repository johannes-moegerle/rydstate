Fitting
=======

.. currentmodule:: rydstate.fit

Fitting the quantum defects of SQDT models to measured energies and intervals of Rydberg levels.
The fittable models themselves are part of :mod:`rydstate.qdt`, see :class:`~rydstate.qdt.ModelSQDTFit`.

.. autosummary::
    :toctree: ../_autosummary/

    fit_sqdt_models
    Measurement
    SQDTLevel
    FitResult

Fits of species
---------------

The quantum defects of the species are fitted with fit config files, which describe the datasets and all choices
of the fit, see :mod:`rydstate.fit.fit_config` and the ``fits/`` directory of the repository.
A fit can be run (or checked) from the terminal:

.. code-block:: bash

    fit_sqdt fits/cesium/fit_config.json --dry-run

.. autosummary::
    :toctree: ../_autosummary/

    fit_config
    fit_from_config
    write_fit
    check_fit
    ConfigFit
    load_measurements
    load_metadata
