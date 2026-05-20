from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from rydstate import __version__
from rydstate.angular.wigner_symbols import calc_wigner_3j
from rydstate.basis.basis_sqdt import BasisSQDT
from rydstate.generate_database.generate_matrix_elements_table import (
    _calc_radial_matrix_element_cached,
    calc_reduced_angular_matrix_element_cached,
    generate_matrix_elements_tables,
    get_radial_state_cached,
)
from rydstate.generate_database.generate_misc_table import generate_wigner_table
from rydstate.generate_database.generate_states_table import generate_states_table

logger = logging.getLogger(__name__)


DATABASE_SQL_FILE = Path(__file__).parent / "database.sql"


def create_tables_for_one_species(
    species_name: str,
    n_min: int,
    n_max: int,
    max_delta_n: float = np.inf,
    all_n_up_to: float = np.inf,
) -> None:
    """Create the database tables for a given species in the current directory."""
    logger.info("Start creating database for %s", species_name)
    logger.info("n-min=%d, n-max=%d", n_min, n_max)
    logger.info("max_delta_n=%s, all_n_up_to=%s", max_delta_n, all_n_up_to)
    logger.info("rydstate.__version__=%s", __version__)

    # create the database and populate the states and matrix elements tables
    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript(DATABASE_SQL_FILE.read_text(encoding="utf-8"))
        basis = BasisSQDT(species_name, n=(n_min, n_max), coupling_scheme="LS")
        generate_states_table(basis, conn)
        generate_matrix_elements_tables(basis, conn, max_delta_n, all_n_up_to)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    # convert the tables to parquet files
    with sqlite3.connect(db_file) as conn:
        tables: pd.DataFrame = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        for tkey in tables.to_numpy().flatten():
            table = pd.read_sql_query(f"SELECT * FROM {tkey}", conn)  # noqa: S608
            if len(table) == 0:
                continue
            if tkey == "states":
                table = table.astype({"is_j_total_momentum": bool, "is_calculated_with_mqdt": bool})

            parquet_file = Path(f"{tkey}.parquet")
            table.to_parquet(parquet_file, index=False, compression="zstd")

            logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
            if logging.getLogger().isEnabledFor(logging.INFO):
                table.info(verbose=True)
                with Path("log").open("a") as buf:
                    table.info(buf=buf)

    logger.info(
        "calc_reduced_angular_matrix_element_cached: %s", calc_reduced_angular_matrix_element_cached.cache_info()
    )
    logger.info("_calc_radial_matrix_element_cached: %s", _calc_radial_matrix_element_cached.cache_info())
    logger.info("get_radial_state_cached: %s", get_radial_state_cached.cache_info())


def create_tables_for_misc(f_max: float, kappa_max: int = 3) -> None:
    """Create misc databases, i.e. the wigner table in the current directory."""
    logger.info("Start creating misc database")
    logger.info("f_max=%s", f_max)
    logger.info("kappa_max=%d", kappa_max)
    logger.info("rydstate.__version__=%s", __version__)

    # create the database and populate the wigner table
    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript(DATABASE_SQL_FILE.read_text(encoding="utf-8"))
        generate_wigner_table(f_max, kappa_max, conn)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    # convert the table to a parquet file
    with sqlite3.connect(db_file) as conn:
        parquet_file = Path("wigner.parquet")
        table = pd.read_sql_query("SELECT * FROM wigner", conn)
        table.to_parquet(parquet_file, index=False, compression="zstd")

        logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
        if logging.getLogger().isEnabledFor(logging.INFO):
            table.info(verbose=True)
            with Path("log").open("a") as buf:
                table.info(buf=buf)

    logger.info("calc_wigner_3j: %s", calc_wigner_3j.cache_info())
