from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

from rydstate.species.mqdt_model import MQDTModel
from rydstate.species.utils import calc_energy_from_nu, calc_polynomial, check_expansion_coefficients

if TYPE_CHECKING:
    from rydstate.species.mqdt import MQDT
    from rydstate.species.utils import ExpansionCoefficients
    from rydstate.units import NDArray


class KMatrixModel(MQDTModel):
    r"""MQDT model parametrized directly by the K-matrix in the outer channel frame (reactance matrix approach).

    The model is defined by the real symmetric K-matrix (reactance matrix) in the frame of the outer channels,
    see e.g. C. L. Vaillant et al., J. Phys. B 47, 155001 (2014).
    Each matrix element is given as a polynomial:

    .. math::
            K_{ij}(\epsilon) = K_{ij}^{(0)} + K_{ij}^{(1)} \epsilon + K_{ij}^{(2)} \epsilon^2 + \dots

    where :math:`\epsilon` is the dimensionless energy variable:

    .. math::
        \epsilon = \frac{I_{\text{ref}} - E}{I_{\text{ref}}} = \frac{Z^2 R_M}{I_{\text{ref}} \nu^2}

    with :math:`I_{\text{ref}}` the reference ionization threshold of the MQDT model
    (see :attr:`~rydstate.species.mqdt.MQDT.reference_ionization_threshold_au`) and :math:`E` the energy of the state,
    both measured from the ground state of the atom.
    """

    k_matrix: ClassVar[list[tuple[int, int, ExpansionCoefficients]]]
    """Elements of the symmetric K-matrix in the outer channel frame.

    Each entry is a tuple (i, j, coefficients) with the indices i <= j of the involved outer channels and
    coefficients the list of expansion coefficients [K⁽⁰⁾, K⁽¹⁾, ...] of the polynomial in the energy variable epsilon
    (a constant element is a single element list).
    Only the upper triangle needs to be given, the lower triangle follows from the symmetry of the K-matrix.
    Elements not given are zero."""

    def __init__(self, mqdt: MQDT) -> None:
        super().__init__(mqdt)

        n = len(self.outer_channels)
        seen: set[tuple[int, int]] = set()
        for i, j, coefficients in self.k_matrix:
            if not (0 <= i <= j < n):
                raise ValueError(f"{self.full_name}: invalid K-matrix indices ({i}, {j}), must be 0 <= i <= j < {n}.")
            if (i, j) in seen:
                raise ValueError(f"{self.full_name}: K-matrix element ({i}, {j}) is given more than once.")
            seen.add((i, j))
            check_expansion_coefficients(coefficients, f"{self.full_name}: k_matrix element ({i}, {j})")
        missing = [i for i in range(n) if (i, i) not in seen]
        if missing:
            raise ValueError(f"{self.full_name}: diagonal K-matrix elements {missing} are not given.")

    def calc_energy_variable(self, nu: float) -> float:
        r"""Return the dimensionless energy variable epsilon, in which the K-matrix elements are expanded.

        .. math::
            \epsilon = \frac{I_{\text{ref}} - E}{I_{\text{ref}}} = \frac{Z^2 R_M}{I_{\text{ref}} \nu^2}

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            The energy variable epsilon (positive for bound states, zero at the reference ionization threshold).

        """
        # we calculate the binding energy directly from nu (and dont use calc_energy_au) to avoid numerical issues
        binding_energy_au = calc_energy_from_nu(
            self.element_properties.reduced_mass_au, nu, self.element_properties.net_charge
        )
        return -binding_energy_au / self.mqdt.reference_ionization_threshold_au

    def calc_k_matrix(self, nu: float) -> NDArray:
        r"""Return the K-matrix in the outer channel frame, evaluated at the energy corresponding to nu.

        Args:
            nu: Effective principal quantum number with reference to the reference ionization threshold.

        Returns:
            K-matrix in the outer channel frame, K = tan(\pi \mu).

        """
        epsilon = self.calc_energy_variable(nu)
        n = len(self.outer_channels)
        kmat = np.zeros((n, n))
        for i, j, coefficients in self.k_matrix:
            value = calc_polynomial(epsilon, coefficients)
            kmat[i, j] = value
            kmat[j, i] = value
        return kmat
