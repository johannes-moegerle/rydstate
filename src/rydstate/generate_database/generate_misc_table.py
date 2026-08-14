from __future__ import annotations

import logging

import numpy as np

from rydstate.angular.wigner_symbols import calc_wigner_3j_with_symmetries

logger = logging.getLogger(__name__)


COLUMNS: dict[str, type] = {
    "f_initial": float,
    "f_final": float,
    "m_initial": float,
    "m_final": float,
    "kappa": int,
    "q": int,
    "val": float,
}


def generate_wigner_table(f_max: float, kappa_max: int) -> dict[str, list[float | int]]:
    """Calculate the wigner table with data for all wigner symbols up to f_max and kappa_max."""
    wigner_data: list[tuple[float | int, ...]] = []
    for start_f_max in [0, 0.5]:  # for better caching
        for kappa in range(kappa_max + 1):
            for f_initial in _get_range(start_f_max, f_max):
                for f_final in _get_range(max(start_f_max, f_initial - kappa), f_initial + kappa):
                    for q in range(-kappa, kappa + 1):
                        for m_initial in _get_range(-f_initial, f_initial):
                            m_final = m_initial + q
                            if not -f_final <= m_final <= f_final:
                                continue
                            wigner = calc_wigner_3j_with_symmetries(f_final, kappa, f_initial, -m_final, q, m_initial)
                            wigner *= (-1) ** (f_final - m_final)
                            if wigner == 0:
                                continue
                            wigner_data.append((f_initial, f_final, m_initial, m_final, kappa, q, wigner))

    assert len(wigner_data) == 0 or len(COLUMNS) == len(wigner_data[0])
    logger.info("Created the 'wigner' table (%s rows)", len(wigner_data))

    return {column: [dtype(row[i]) for row in wigner_data] for i, (column, dtype) in enumerate(COLUMNS.items())}


def _get_range(vmin: float, vmax: float) -> range | np.ndarray:
    if vmin % 1 == 0:
        return range(int(vmin), int(vmax) + 1)
    return np.arange(vmin, vmax + 0.5)
