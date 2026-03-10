from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from functools import cache, cached_property
from typing import TYPE_CHECKING, ClassVar, TypeVar, overload

from rydstate.units import rydberg_constant, ureg

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpec, Self

    from rydstate.radial.model import PotentialType
    from rydstate.units import PintFloat

    P = ParamSpec("P")
    R = TypeVar("R")

    def cache(func: Callable[P, R]) -> Callable[P, R]: ...  # type: ignore [misc]


logger = logging.getLogger(__name__)


class SpeciesObject(ABC):
    """Abstract base class for all species objects.

    For the electronic ground state configurations and sorted shells,
    see e.g. https://www.webelements.com/atoms.html

    """

    name: ClassVar[str]
    """The name of the atomic species."""
    Z: ClassVar[int]
    """Atomic number of the species."""
    i_c: ClassVar[float | None] = None
    """Nuclear spin, (default None to ignore hyperfine structure, will be treated like i_c = 0)."""
    number_valence_electrons: ClassVar[int]
    """Number of valence electrons (i.e. 1 for alkali atoms and 2 for alkaline earth atoms)."""

    _corrected_rydberg_constant: tuple[float, float | None, str]
    r"""Corrected Rydberg constant stored as (value, uncertainty, unit)"""

    potential_type_default: PotentialType | None = None
    """Default potential type to use for this species. If None, the potential type must be specified explicitly.
    In general, it looks like marinescu_1993 is better for alkali atoms, and fei_2009 is better for alkaline earth atoms
    """

    # Model Potential Parameters for marinescu_1993
    alpha_c_marinescu_1993: ClassVar[float]
    """Static dipole polarizability in atomic units (a.u.), used for the parametric model potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """
    r_c_dict_marinescu_1993: ClassVar[dict[int, float]]
    """Cutoff radius {l: r_c} to truncate the unphysical short-range contribution of the polarization potential.
    See also: Phys. Rev. A 49, 982 (1994)
    """
    model_potential_parameter_marinescu_1993: ClassVar[dict[int, tuple[float, float, float, float]]]
    """Parameters {l: (a_1, a_2, a_3, a_4)} for the parametric model potential.
    See also: M. Marinescu, Phys. Rev. A 49, 982 (1994), https://journals.aps.org/pra/abstract/10.1103/PhysRevA.49.982
    """

    # Model Potential Parameters for fei_2009
    model_potential_parameter_fei_2009: tuple[float, float, float, float]
    """Parameters (delta, alpha, beta, gamma) for the new four-parameter potential, used in the model potential
    defined in: Y. Fei et al., Chin. Phys. B 18, 4349 (2009), https://iopscience.iop.org/article/10.1088/1674-1056/18/10/025
    """

    @property
    def i_c_number(self) -> float:
        """Return a numerical value for i_c, i.e. either i_c or 0 if i_c is None."""
        return self.i_c if self.i_c is not None else 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @classmethod
    @cache
    def from_name(cls: type[Self], name: str) -> Self:
        """Create an instance of the species class from the species name.

        This method searches through all subclasses of SpeciesObject until it finds one with a matching species name.
        This approach allows for easy extension of the library with new species.
        A user can even subclass SpeciesObjectSQDT in his code (without modifying the rydstate library),
        e.g. `class CustomRubidium(SpeciesObjectSQDT): name = "Custom_Rb" ...`
        and then use the new species by calling RydbergStateSQDTAlkali("Custom_Rb", ...)

        Args:
            name: The species name (e.g. "Rb").

        Returns:
            An instance of the corresponding species class.

        """
        concrete_subclasses = cls._get_concrete_subclasses()
        for subclass in concrete_subclasses:
            if subclass.name == name:
                return subclass()
        raise ValueError(
            f"Unknown species name: {name}. Available species: {[subclass.name for subclass in concrete_subclasses]}"
        )

    @classmethod
    def _get_concrete_subclasses(cls: type[Self]) -> list[type[Self]]:
        subclasses = []
        for subclass in cls.__subclasses__():
            if not inspect.isabstract(subclass) and hasattr(subclass, "name"):
                subclasses.append(subclass)
            subclasses.extend(subclass._get_concrete_subclasses())  # noqa: SLF001
        return subclasses

    @classmethod
    def get_available_species(cls) -> list[str]:
        """Get a list of all available species names in the library.

        This method returns a list of species names for all concrete subclasses of SpeciesObject.

        Returns:
            List of species names.

        """
        return sorted([subclass.name for subclass in cls._get_concrete_subclasses()])

    @property
    @abstractmethod
    def reference_ionization_energy_au(self) -> float: ...

    @overload
    def get_corrected_rydberg_constant(self, unit: None = None) -> PintFloat: ...

    @overload
    def get_corrected_rydberg_constant(self, unit: str) -> float: ...

    def get_corrected_rydberg_constant(self, unit: str | None = "hartree") -> PintFloat | float:
        r"""Return the corrected Rydberg constant in the desired unit.

        The corrected Rydberg constant is defined as

        .. math::
            R_M = R_\infty \frac{m_{Core}}{m_{Core} + m_e}

        where :math:`R_\infty` is the Rydberg constant for infinite nuclear mass,
        :math:`m_{Core}` is the mass of the core,
        and :math:`m_e` is the mass of the electron.

        Args:
            unit: Desired unit for the corrected Rydberg constant. Default is atomic units "hartree".

        Returns:
            Corrected Rydberg constant in the desired unit.

        """
        corrected_rydberg_constant: PintFloat = ureg.Quantity(
            self._corrected_rydberg_constant[0], self._corrected_rydberg_constant[2]
        )
        corrected_rydberg_constant = corrected_rydberg_constant.to("hartree", "spectroscopy")
        if unit is None:
            return corrected_rydberg_constant
        if unit == "a.u.":
            return corrected_rydberg_constant.magnitude
        return corrected_rydberg_constant.to(unit, "spectroscopy").magnitude

    @cached_property  # don't remove this caching without benchmarking it!!!
    def reduced_mass_au(self) -> float:
        r"""The reduced mass mu in atomic units.

        The reduced mass in atomic units :math:`\mu / m_e` is given by

        .. math::
            \frac{\mu}{m_e} = \frac{m_{Core}}{m_{Core} + m_e}

        We calculate the reduced mass via the corrected Rydberg constant

        .. math::
            \frac{\mu}{m_e} = \frac{R_M}{R_\infty}

        """
        return self.get_corrected_rydberg_constant("hartree") / rydberg_constant.to("hartree").m
