from __future__ import annotations

from typing import TYPE_CHECKING

from rydstate.qdt.sqdt.model_sqdt import ModelSQDT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self


class ModelSQDTFit(ModelSQDT):
    """SQDT model, whose parameters can be changed, e.g. to fit them to data.

    The models of the SQDT model sets of the species are shared, e.g. by all states of a species,
    so their parameters cannot be changed (:attr:`~rydstate.qdt.ModelSQDT.quantum_defects` is a tuple).
    Instead, a fit uses an independent copy of a model, which is created via :meth:`from_model`:

    .. code-block:: python

        model_fit = ModelSQDTFit.from_model(model)
        model_fit.set_parameters([3.13, 0.18])

    Like any other SQDT model, a fitted model is stored as a :class:`~rydstate.qdt.ModelSQDT`,
    see :meth:`~rydstate.qdt.ModelSQDT.to_dict`.
    """

    @classmethod
    def from_model(cls, model: ModelSQDT) -> Self:
        """Create an independent copy of the given model, whose parameters can be changed.

        Args:
            model: The model to copy.

        """
        return cls(
            model.species,
            model.channel,
            model.quantum_defects,
            reference=model.reference,
            nu_range=model.nu_range,
        )

    @property
    def number_of_parameters(self) -> int:
        """Number of free parameters of the model, see :meth:`get_parameters`."""
        return len(self.quantum_defects)

    def get_parameters(self) -> list[float]:
        """Return the free parameters of the model.

        The parameters are the expansion coefficients [d0, d2, d4, ...] of the quantum defect.
        Every given order is a free parameter, i.e. a model initialized with [d0, 0] has two parameters.

        Returns:
            A (new) list of the parameters, see also :meth:`set_parameters`.

        """
        return list(self.quantum_defects)

    def set_parameters(self, parameters: Sequence[float]) -> None:
        """Set the free parameters of the model, see :meth:`get_parameters`.

        Args:
            parameters: The new parameters, i.e. the expansion coefficients [d0, d2, d4, ...] of the quantum defect.
                Their number must be :attr:`number_of_parameters`.

        """
        self.quantum_defects = tuple(parameters)
