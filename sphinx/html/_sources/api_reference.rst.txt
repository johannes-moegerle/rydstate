API Reference
=============


The RydState python API can be accessed via the ``rydstate`` module by

.. code-block:: python

    import rydstate


All the available classes, methods and functions are documented below:

.. currentmodule:: rydstate

**Rydberg States**

.. autosummary::
    :toctree: _autosummary/

       RydbergStateSQDT
       RydbergStateSQDTAlkali
       RydbergStateSQDTAlkalineLS
       RydbergStateSQDTAlkalineJJ
       RydbergStateSQDTAlkalineFJ

**Rydberg Basis**

.. autosummary::
    :toctree: _autosummary/

       BasisSQDTAlkali
       BasisSQDTAlkalineLS
       BasisSQDTAlkalineJJ
       BasisSQDTAlkalineFJ

**Angular module**

.. autosummary::
    :toctree: _autosummary/

       angular.AngularKetLS
       angular.AngularKetJJ
       angular.AngularKetFJ
       angular.AngularState
       angular.utils


**Radial module**

.. autosummary::
    :toctree: _autosummary/

       radial.RadialKet
       radial.Wavefunction
       radial.Model
       radial.numerov

**Species module and parameters**

.. autosummary::
    :toctree: _autosummary/

       species.SpeciesObjectSQDT
       species.HydrogenTextBook
       species.Hydrogen
       species.Lithium
       species.Sodium
       species.Potassium
       species.Rubidium
       species.Cesium
       species.Strontium87
       species.Strontium88
       species.Ytterbium171
       species.Ytterbium173
       species.Ytterbium174
