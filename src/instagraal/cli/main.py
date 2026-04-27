"""Click CLI for the ``instagraal`` scaffolding command."""

import logging
import pathlib

import click

from ..version import __version__ as VERSION_NUMBER


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION_NUMBER, "-V", "--version")
@click.argument(
    "hic_folder",
    type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "reference_fa",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    metavar="REFERENCE.FA",
)
@click.option(
    "-o",
    "--output-dir",
    "output_folder",
    default="out",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Directory where output files will be written.",
)
@click.option(
    "-l",
    "--level",
    default=4,
    show_default=True,
    type=int,
    help=(
        "Level (resolution) of the contact map. "
        "Increasing level by one means a threefold smaller resolution "
        "but also a threefold faster computation time."
    ),
)
@click.option(
    "-n",
    "--cycles",
    default=100,
    show_default=True,
    type=int,
    help=(
        "Number of iterations to perform for each bin "
        "(row/column of the contact map). A high number of cycles has "
        "diminishing returns but there is a necessary minimum for "
        "assembly convergence."
    ),
)
@click.option(
    "-c",
    "--coverage-std",
    default=1.0,
    show_default=True,
    type=float,
    help=(
        "Number of standard deviations below the mean coverage, below which fragments should be filtered out prior to binning."
    ),
)
@click.option(
    "-N",
    "--neighborhood",
    default=5,
    show_default=True,
    type=int,
    help="Number of neighbors to sample for potential mutations for each bin.",
)
@click.option(
    "--device",
    default=0,
    show_default=True,
    type=int,
    help="If multiple graphic cards are available, select a specific device (numbered from 0).",
)
@click.option(
    "-C",
    "--circular",
    is_flag=True,
    default=False,
    help="Indicates genome is circular.",
)
@click.option(
    "-b",
    "--bomb",
    is_flag=True,
    default=False,
    help="Explode the genome prior to scaffolding.",
)
@click.option(
    "--pyramid-only",
    is_flag=True,
    default=False,
    help="Only build multi-resolution contact maps (pyramids) and skip scaffolding.",
)
@click.option(
    "--save-pickle",
    is_flag=True,
    default=False,
    help="Dump all info from the instaGRAAL run into a pickle (for development/introspection).",
)
@click.option(
    "--save-matrix",
    is_flag=True,
    default=False,
    help="Save a preview of the contact map after each cycle.",
)
@click.option(
    "--simple",
    is_flag=True,
    default=False,
    help="Only perform operations at the edge of the contigs.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Only display warnings and errors as outputs.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Display debug information. Overrides --quiet. For development purposes only.",
)
def main(
    hic_folder: pathlib.Path,
    reference_fa: pathlib.Path,
    output_folder: pathlib.Path | None,
    level: int,
    cycles: int,
    coverage_std: float,
    neighborhood: int,
    device: int,
    circular: bool,
    bomb: bool,
    pyramid_only: bool,
    save_pickle: bool,
    save_matrix: bool,
    simple: bool,
    quiet: bool,
    debug: bool,
) -> None:
    """Large genome reassembly based on Hi-C data.

    HIC_FOLDER  Directory containing the Hi-C contact map files produced by
    instagraal-pre.

    REFERENCE.FA  Reference genome in FASTA format.
    """
    from .. import log
    from ..instagraal import run_instagraal

    log_level = logging.INFO
    if quiet:
        log_level = logging.WARNING
    if debug:
        log_level = logging.DEBUG

    log.logger.setLevel(log_level)
    log.CURRENT_LOG_LEVEL = log_level

    run_instagraal(
        hic_folder=hic_folder,
        reference_fa=reference_fa,
        output_folder=output_folder,
        level=level,
        cycles=cycles,
        coverage_std=coverage_std,
        neighborhood=neighborhood,
        device=device,
        circular=circular,
        bomb=bomb,
        pyramid_only=pyramid_only,
        save_pickle=save_pickle,
        save_matrix=save_matrix,
        simple=simple,
    )
