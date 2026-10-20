from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal

from rydstate.angular.angular_ket import AngularKetBase
from rydstate.angular.utils import is_not_set
from rydstate.qdt.model import Model
from rydstate.qdt.utils import calc_modified_ritz_formula, check_expansion_coefficients

if TYPE_CHECKING:
    from rydstate.angular.angular_ket import AngularKetLS
    from rydstate.angular.utils import AllKnown
    from rydstate.qdt.utils import ExpansionCoefficients


class ModelSQDT(Model):
    r"""SQDT model describing the quantum defects of a single channel of a species.

    In contrast to :class:`~rydstate.qdt.ModelMQDT`, the parameters of an SQDT model are not defined
    in a subclass, but simply given when constructing the model, e.g.

    .. code-block:: python

        ModelSQDT("Rb", "ns_1/2", [3.1311807, 0.1787], reference="Phys. Rev. A 83, 052515 (2011)")

    The quantum defect of the channel is given by the modified Ritz formula

    .. math::
        \delta(n) = \delta_0 + \frac{\delta_2}{(n - \delta_0)^2} + \frac{\delta_4}{(n - \delta_0)^4} + ...

    see :meth:`calc_quantum_defect`.
    """

    def __init__(
        self,
        species: str,
        channel: AngularKetBase[Any] | str,
        quantum_defects: ExpansionCoefficients,
        reference: str | tuple[str, ...] | None = None,
        nu_range: tuple[float, float | Literal["inf"]] = (0, math.inf),
    ) -> None:
        """Initialize the SQDT model.

        Args:
            species: The species for which the model is defined.
            channel: The channel described by the model, either as angular ket or as its short label
                (e.g. "ns_1/2" or "5s np 3P1",
                see :meth:`~rydstate.angular.angular_ket.AngularKetBase.from_short_label`).
                The channel must be a single LS ket, i.e. have well defined l_r, j_tot and s_tot,
                and its magnetic quantum number m must be NotSet.
            quantum_defects: The expansion coefficients [d0, d2, d4, ...] of the quantum defect,
                see :meth:`calc_quantum_defect`.
            reference: Reference for the model, e.g., a publication doi where the quantum defects are given.
            nu_range: Range (nu_min, nu_max) of effective principal quantum numbers nu for which the model is valid.
                nu_min must be finite, an infinite nu_max can also be given as the string "inf".
                Default is (0, inf), i.e. the model is valid for all (positive) nu.

        """
        if isinstance(channel, str):
            channel = AngularKetBase.from_short_label(channel, species)
        if channel.coupling_scheme != "LS":
            angular_state = channel.to_state("LS")
            if len(angular_state.kets) != 1:
                raise ValueError(f"The channel {channel} of an SQDT model must be a single LS ket.")
            channel = angular_state.kets[0]
        if not is_not_set(channel.m):
            raise ValueError(f"The magnetic quantum number m of the channel {channel} must be NotSet.")
        if channel.contains_unknown:
            raise ValueError(f"The channel {channel} of an SQDT model must not contain unknown quantum numbers.")

        self.species = species
        self.channel: AngularKetLS[AllKnown] = channel  # type: ignore [assignment]
        self.name = self.channel.get_short_label()
        self.reference = reference
        self.f_tot = self.channel.f_tot
        self.parity = self.channel.parity
        nu_min, nu_max = nu_range
        if nu_max == "inf":
            nu_max = math.inf
        elif isinstance(nu_max, str):
            raise ValueError(f'{self.full_name}: nu_max must be a number or the string "inf", but got {nu_max!r}.')
        if not math.isfinite(nu_min):
            raise ValueError(f"{self.full_name}: nu_min must be finite, but got {nu_min}.")
        self.nu_range = (nu_min, nu_max)
        self.quantum_defects = quantum_defects
        super().__init__()

        check_expansion_coefficients(quantum_defects, f"{self.full_name}: quantum_defects")

    def calc_quantum_defect(self, n: int) -> float:
        r"""Calculate the quantum defect of the state with principal quantum number n via the modified Ritz formula.

        .. math::
            \delta(n) = \delta_0 + \frac{\delta_2}{(n - \delta_0)^2} + \frac{\delta_4}{(n - \delta_0)^4} + ...

        References:
            - On a New Law of Series Spectra, Ritz; DOI: 10.1086/141591, https://ui.adsabs.harvard.edu/abs/1908ApJ....28..237R/abstract
            - Rydberg atoms, Gallagher; DOI: 10.1088/0034-4885/51/2/001, (Eq. 16.19)

        Args:
            n: The principal quantum number of the Rydberg state.

        """
        return calc_modified_ritz_formula(n, self.quantum_defects)

    def calc_nui(self, n: int) -> float:
        """Calculate the effective principal quantum number nui = n - delta(n) of the state with the given n.

        Args:
            n: The principal quantum number of the Rydberg state.

        """
        return n - self.calc_quantum_defect(n)
