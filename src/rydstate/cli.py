from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from rydstate.generate_database.generate_database import (
    create_tables_for_misc,
    create_tables_for_mqdt,
    create_tables_for_sqdt,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the generate_database script."""
    parser = build_parser()
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    try:
        directory = prepare_directory(args)
    except OSError as err:
        # the message is meant for the user, so print it without a traceback
        sys.exit(f"Error: {err}")
    configure_logging(args.log_level, directory, args.warnings_as_exceptions)

    time_start = time.perf_counter()
    if args.mode == "misc":
        create_tables_for_misc(f_max=args.f_max, kappa_max=3)
    elif args.mode == "sqdt":
        create_tables_for_sqdt(
            args.species,
            n=(args.n_min, args.n_max),
            f_tot=get_qn_range(args, "f_tot"),
            l_r=get_qn_range(args, "l_r"),  # type: ignore [arg-type]
            max_delta_nu=args.max_delta_nu,
            all_nu_up_to=args.all_nu_up_to,
        )
    elif args.mode == "mqdt":
        create_tables_for_mqdt(
            args.species,
            nu=(args.nu_min, args.nu_max),
            f_tot=get_qn_range(args, "f_tot"),
            l_r=get_qn_range(args, "l_r"),  # type: ignore [arg-type]
            max_delta_nu=args.max_delta_nu,
            all_nu_up_to=args.all_nu_up_to,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    logger.info("Time taken: %.2f seconds", time.perf_counter() - time_start)
    log_memory_usage()


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with the sqdt, mqdt and misc subcommands."""
    parser = argparse.ArgumentParser(
        description="Generate a database, containing energies and matrix elements, for a given species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  generate_database sqdt Rb --n-max 60\n"
            "  generate_database mqdt Yb174 --nu-max 60\n"
            "  generate_database misc --f-max 10\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="mode", title="modes")

    # arguments shared by all modes
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--directory",
        default=None,
        type=str,
        help="The directory where the database will be saved. "
        "Defaults to a subfolder of database/ derived from the mode and species.",
    )
    common.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO).",
    )
    common.add_argument(
        "--warnings-as-exceptions",
        action="store_true",
        help="Treat warnings in rydstate as exceptions.",
    )
    common.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the folder if it exists and create a new one.",
    )

    # arguments shared by the species modes (sqdt and mqdt)
    species_common = argparse.ArgumentParser(add_help=False)
    species_common.add_argument("species", help="The species name to generate the database for (e.g. Rb).")
    species_common.add_argument(
        "--f-tot-min",
        default=None,
        type=float,
        help="The minimal total angular momentum quantum number f_tot for the states to be included in the database. "
        "Defaults to 0.",
    )
    species_common.add_argument(
        "--f-tot-max",
        default=None,
        type=float,
        help="The maximum total angular momentum quantum number f_tot for the states to be included in the database. "
        "Defaults to inf.",
    )
    species_common.add_argument(
        "--l-r-min",
        default=None,
        type=int,
        help="The minimal orbital angular momentum quantum number l_r for the states to be included in the database. "
        "Defaults to 0.",
    )
    species_common.add_argument(
        "--l-r-max",
        default=None,
        type=int,
        help="The maximum orbital angular momentum quantum number l_r for the states to be included in the database. "
        "Defaults to inf.",
    )
    species_common.add_argument(
        "--max-delta-nu",
        default=float("inf"),
        type=float,
        help="The maximum difference in the effective principal quantum number nu for matrix elements to be included.",
    )
    species_common.add_argument(
        "--all-nu-up-to",
        default=float("inf"),
        type=float,
        help="Calculate all matrix elements where at least one state has an effective principal quantum number nu "
        "smaller than or equal to this value.",
    )

    # sqdt parser
    sqdt_parser = subparsers.add_parser(
        "sqdt",
        parents=[species_common, common],
        help="Generate the database for a species using single-channel quantum defect theory.",
        description="Generate the database for a species using single-channel quantum defect theory. "
        "The basis is defined via the n-range.",
    )
    sqdt_parser.add_argument(
        "--n-min",
        default=1,
        type=int,
        help="The minimal principal quantum number n for the states to be included in the database. "
        "Default 1 will start with the ground state configuration of the specific species (e.g. n=5 for Rb).",
    )
    sqdt_parser.add_argument(
        "--n-max",
        required=True,
        type=int,
        help="The maximum principal quantum number n for the states to be included in the database.",
    )

    # mqdt parser
    mqdt_parser = subparsers.add_parser(
        "mqdt",
        parents=[species_common, common],
        help="Generate the database for a species using multi-channel quantum defect theory.",
        description="Generate the database for a species using multi-channel quantum defect theory. "
        "The basis is defined via the nu-range.",
    )
    mqdt_parser.add_argument(
        "--nu-min",
        default=0,
        type=float,
        help="The minimal effective principal quantum number nu for the states to be included in the database. "
        "Default 0 will include all low lying states.",
    )
    mqdt_parser.add_argument(
        "--nu-max",
        required=True,
        type=float,
        help="The maximum effective principal quantum number nu for the states to be included in the database.",
    )

    # misc parser
    misc_parser = subparsers.add_parser(
        "misc",
        parents=[common],
        help="Generate the misc database tables, which do not depend on a species.",
        description="Generate the misc database tables, which do not depend on a species.",
    )
    misc_parser.add_argument(
        "--f-max",
        required=True,
        type=float,
        help="The maximum angular momentum quantum number f for misc database tables.",
    )

    return parser


def prepare_directory(args: argparse.Namespace) -> Path:
    """Create the (empty) database directory and change into it."""
    if args.directory is not None:
        directory = Path(args.directory)
    elif args.mode == "misc":
        directory = Path("database") / "misc"
    elif args.mode == "sqdt":
        species = args.species.removesuffix("_sqdt")
        folder = f"{species}_sqdt" if any(species.startswith(sp) for sp in ["Sr", "Yb"]) else species
        directory = Path("database") / folder
    elif args.mode == "mqdt":
        species = args.species.removesuffix("_mqdt")
        directory = Path("database") / f"{species}_mqdt"
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    directory = directory.resolve()

    if directory.exists():
        if not args.overwrite:
            raise FileExistsError(f"The folder {directory} already exists. Use --overwrite to overwrite it.")
        check_is_generated_database(directory)
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    os.chdir(directory)
    return directory


def check_is_generated_database(directory: Path) -> None:
    """Raise if the directory contains anything but the files written by a previous run.

    Since --overwrite deletes the whole directory tree, only ever delete a directory that looks like a
    previously generated database, i.e. that solely contains the log file and the parquet tables.
    """
    if not directory.is_dir():
        raise FileExistsError(
            f"Refusing to overwrite {directory}, since it is not a directory. "
            "Delete it manually if this is really what you want."
        )
    unexpected = sorted(
        entry.name
        for entry in directory.iterdir()
        if not (entry.is_file() and (entry.name == "log" or entry.suffix == ".parquet"))
    )
    if unexpected:
        listed = ", ".join(unexpected[:5]) + (", ..." if len(unexpected) > 5 else "")
        raise FileExistsError(
            f"Refusing to overwrite the folder {directory}, since it does not look like a generated database "
            f"(it contains {listed}). Delete it manually if this is really what you want."
        )


def get_qn_range(args: argparse.Namespace, qn: str) -> tuple[float, float] | None:
    """Get the (<qn>_min, <qn>_max) range for the quantum number qn from the parsed arguments.

    Returns None if neither the minimum nor the maximum is given, i.e. all values of qn are included.
    """
    qn_min, qn_max = getattr(args, f"{qn}_min"), getattr(args, f"{qn}_max")
    if qn_min is None and qn_max is None:
        return None
    return (
        qn_min if qn_min is not None else 0,
        qn_max if qn_max is not None else float("inf"),
    )


def log_memory_usage() -> None:
    """Log the peak and current memory usage (resident set size) of this process."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        # the resource module is not available on some platforms (e.g. Windows), so we skip the peak memory logging
        pass
    else:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kilobytes on Linux, bytes on macOS
        peak_memory = ru_maxrss * (1e-6 if sys.platform == "darwin" else 1e-3)
        logger.info("Peak memory usage: %.2f megabytes", peak_memory)

    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        current_memory = resident_pages * os.sysconf("SC_PAGE_SIZE") * 1e-6
        logger.info("Memory usage at end of run: %.2f megabytes", current_memory)


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
