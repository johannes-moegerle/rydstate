# The species data subclasses the QDT classes defined here, and the QDT classes need the species data at runtime.
# Importing the species module first ensures, that the QDT modules are fully initialized
# before the species data is loaded (and not only partially, if rydstate.qdt is imported first).
from rydstate import species  # noqa: F401  # isort: skip  # must be imported first

from rydstate.qdt.eigen_channel_model import EigenChannelModel, TrivialModel
from rydstate.qdt.k_matrix_model import KMatrixModel
from rydstate.qdt.mqdt import MQDT, get_mqdt
from rydstate.qdt.mqdt_model import MQDTModel, ScaledOffDiagonalModel
from rydstate.qdt.osqdt import OSQDT, OSQDTModel
from rydstate.qdt.sqdt import SQDT, get_sqdt

__all__ = [
    "MQDT",
    "OSQDT",
    "SQDT",
    "EigenChannelModel",
    "KMatrixModel",
    "MQDTModel",
    "OSQDTModel",
    "ScaledOffDiagonalModel",
    "TrivialModel",
    "get_mqdt",
    "get_sqdt",
]
