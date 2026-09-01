from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from rydstate.angular.utils import NotSet, is_not_set


def get_m_range(f_tot: float, m_range: tuple[float, float] | NotSet | None) -> list[NotSet] | list[float]:
    if is_not_set(m_range):
        return [NotSet]  # type: ignore[return-value]
    if m_range is None:
        m_range = (-f_tot, f_tot)

    allowed_m_values = np.arange(-f_tot, f_tot + 1)
    return [float(m) for m in allowed_m_values if m_range[0] <= m <= m_range[1]]


def is_allowed_qn(qn_range: tuple[float, float] | None, qn: float, delta: float = 1e-6) -> bool:
    if qn_range is None:
        return True
    if isinstance(qn_range, Sequence) and len(qn_range) == 2:
        return qn_range[0] - delta <= qn <= qn_range[1] + delta
    raise ValueError(f"Invalid qn_range: {qn_range}. Must be None or a tuple of two numbers.")
