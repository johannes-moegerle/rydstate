from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from rydstate.angular.wigner_symbols import calc_wigner_3j_with_symmetries

if TYPE_CHECKING:
    import sqlite3


logger = logging.getLogger(__name__)


COLUMNS = ["f_initial", "f_final", "m_initial", "m_final", "kappa", "q", "val"]


def generate_wigner_table(
    f_max: float, kappa_max: int, conn: sqlite3.Connection | None = None
) -> list[tuple[float | int, ...]]:
    """Populate the wigner table with data for all wigner symbols up to f_max and kappa_max."""
    wigner_data: list[tuple[float | int, ...]] = []
    for start_f_max in [0, 0.5]:  # for better caching
        for kappa in range(kappa_max + 1):
            for f_initial in np.arange(start_f_max, f_max + 0.5, 1):
                for f_final in np.arange(np.max([f_initial % 1, f_initial - kappa]), f_initial + kappa + 1):
                    for q in range(-kappa, kappa + 1):
                        for m_initial in np.arange(-f_initial, f_initial + 1):
                            m_final = m_initial + q
                            if not -f_final <= m_final <= f_final:
                                continue
                            wigner = calc_wigner_3j_with_symmetries(f_final, kappa, f_initial, -m_final, q, m_initial)
                            wigner *= (-1) ** (f_final - m_final)
                            if wigner == 0:
                                continue
                            wigner_data.append((f_initial, f_final, m_initial, m_final, kappa, q, wigner))

    assert len(wigner_data) == 0 or len(COLUMNS) == len(wigner_data[0])

    if conn is not None:
        stmt = f"INSERT INTO wigner ({', '.join(COLUMNS)}) VALUES ({', '.join(['?'] * len(COLUMNS))})"  # noqa: S608
        conn.executemany(stmt, wigner_data)
        num_rows = conn.execute("SELECT COUNT(*) FROM wigner").fetchone()[0]
        logger.info("Created the 'wigner' table (%s rows)", num_rows)

    return wigner_data
