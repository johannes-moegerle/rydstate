from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rydstate.qdt.sqdt.model_sqdt import ModelSQDT

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
            n_range=model.n_range,
            fit_info=None,
        )

    def to_model(self, fit_info: Mapping[str, Any] | None = None) -> ModelSQDT:
        """Return an independent model with the present parameters, e.g. to store or use a fitted model.

        The model is a :class:`~rydstate.qdt.ModelSQDT` with the same species, channel, reference and n range.
        A subclass whose parameters are not the quantum defects of :class:`~rydstate.qdt.ModelSQDT`
        (another parametrization) must override this method.

        Args:
            fit_info: Information on how the parameters were fitted, see :attr:`~rydstate.qdt.ModelSQDT.fit_info`.
                Default is None.

        """
        return ModelSQDT(
            self.species,
            self.channel,
            self.quantum_defects,
            reference=self.reference,
            n_range=self.n_range,
            fit_info=fit_info,
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

    def calc_nui_gradient(self, n: int) -> list[float]:
        r"""Return the derivatives of nui = n - delta(n) with respect to the parameters, see :meth:`get_parameters`.

        With :math:`x = 1/(n - \delta_0)^2` and :math:`\delta(n) = \delta_0 + \sum_{k \geq 1} \delta_{2k} x^k`
        (see :meth:`~rydstate.qdt.ModelSQDT.calc_quantum_defect`), the derivatives are
        :math:`\partial \delta / \partial \delta_{2k} = x^k` and
        :math:`\partial \delta / \partial \delta_0 = 1 + x' \sum_{k \geq 1} k \delta_{2k} x^{k-1}`
        with :math:`x' = \partial x / \partial \delta_0 = 2/(n - \delta_0)^3`.

        Args:
            n: The principal quantum number of the Rydberg state.

        Returns:
            The list of the derivatives d nui / d parameter, in the order of :meth:`get_parameters`.

        """
        d0, *higher_orders = self.quantum_defects
        x = 1.0 / (n - d0) ** 2
        dx_dd0 = 2.0 / (n - d0) ** 3
        ddelta_dd0 = 1.0 + dx_dd0 * sum(k * p * x ** (k - 1) for k, p in enumerate(higher_orders, start=1))
        ddelta = [ddelta_dd0] + [x**k for k in range(1, len(self.quantum_defects))]
        return [-value for value in ddelta]

    def calc_binding_energy_gradient(self, n: int) -> list[float]:
        r"""Return the derivatives of the binding energy with respect to the parameters, see :meth:`get_parameters`.

        The binding energy :math:`B = -Z^2 \mu / (2 \nu_i^2)` (see :meth:`calc_binding_energy_au`) has the derivatives
        :math:`\partial B / \partial p = Z^2 \mu / \nu_i^3 \, \partial \nu_i / \partial p`,
        with :math:`\partial \nu_i / \partial p` from :meth:`calc_nui_gradient`.
        The ionization threshold does not depend on the parameters, so these are also the derivatives of the energy
        (see :meth:`calc_energy_au`).

        Args:
            n: The principal quantum number of the Rydberg state.

        Returns:
            The list of the derivatives d B / d parameter in atomic units, in the order of :meth:`get_parameters`.

        """
        properties = self.element_properties
        dbinding_dnui = properties.net_charge**2 * properties.reduced_mass_au / self.calc_nui(n) ** 3
        return [dbinding_dnui * value for value in self.calc_nui_gradient(n)]
