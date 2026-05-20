from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from pathlib import Path

from rydstate.generate_database.generate_database import create_tables_for_misc, create_tables_for_one_species

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the generate_database script."""
    parser = argparse.ArgumentParser(
        description="Generate a database, containing energies and matrix elements, for a given species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  generate_database Rb --log-level INFO\n"),
    )
    parser.add_argument("species", help="The species name to generate the database for.")
    parser.add_argument(
        "--n-min",
        default=None,
        type=int,
        help="The minimal principal quantum number n for the states to be included in the database. "
        "This is used for species, where the low lying states do not converge nicely, so we exclude those states. "
        "Default 1 will start with the ground state configuration of the specific species (e.g. n=5 for Rb).",
    )
    parser.add_argument(
        "--n-max",
        default=None,
        type=int,
        help="The maximum principal quantum number n for the states to be included in the database.",
    )
    parser.add_argument(
        "--f-max",
        default=None,
        type=float,
        help="The maximum angular momentum quantum number f for misc database tables.",
    )
    parser.add_argument(
        "--directory",
        default=None,
        type=str,
        help="The directory where the database will be saved.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--warnings-as-exceptions",
        action="store_true",
        help="Treat warnings in rydstate as exceptions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the species folder if it exists and create a new one.",
    )

    args = parser.parse_args()
    if args.species == "misc":
        if args.n_min is not None or args.n_max is not None:
            parser.error("--n-min and --n-max are only valid when generating a species database.")
    elif args.f_max is not None:
        parser.error("--f-max is only valid when generating the misc database.")

    directory = Path(args.directory) if args.directory is not None else Path("database") / args.species
    directory = directory.resolve()
    if directory.exists():
        if args.overwrite:
            shutil.rmtree(directory)
        else:
            raise FileExistsError(f"The folder {directory} already exists. Use --overwrite to overwrite it.")
    directory.mkdir(parents=True)
    os.chdir(directory)

    configure_logging(args.log_level, directory, args.warnings_as_exceptions)

    time_start = time.perf_counter()
    if args.species == "misc":
        if args.f_max is None:
            parser.error("--f-max is required when generating the misc database.")
        create_tables_for_misc(f_max=args.f_max, kappa_max=3)
    else:
        n_min = args.n_min if args.n_min is not None else 1
        if args.n_max is None:
            parser.error("--n-max is required when generating the misc database.")
        create_tables_for_one_species(args.species, n_min, args.n_max)
    logger.info("Time taken: %.2f seconds", time.perf_counter() - time_start)


def configure_logging(
    log_level: str,
    log_directory: Path | None = None,
    warnings_as_exceptions: bool = False,
) -> None:
    """Initialize the logger."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(getattr(logging, log_level.upper()))

    stream_formatter = logging.Formatter("%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    if log_directory is not None:
        file_formatter = logging.Formatter("%(levelname)s: %(message)s")
        file_handler = logging.FileHandler(log_directory / "log", delay=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    if warnings_as_exceptions:
        root_logger.addHandler(WarningsAsExceptionsHandler())


class WarningsAsExceptionsHandler(logging.Handler):
    """Custom logging handler to raise exceptions for log warnings and above."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            raise RuntimeError(record.getMessage())


if __name__ == "__main__":
    main()
