"""Click CLI for the ``instagraal-refine`` command."""

import pathlib

import click

from ..refine import (
    DEFAULT_BINSIZE,
    DEFAULT_MIN_CONTIG_LENGTH,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SCORE,
    DEFAULT_WINDOW,
    run_refine,
)
from ..version import __version__ as VERSION_NUMBER


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION_NUMBER, "-V", "--version")
@click.option(
    "-i",
    "--info-frags",
    "info_frags",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Layout to check and refine: new_info_frags.txt from instagraal-polish (or instagraal's info_frags.txt).",
)
@click.option(
    "-c",
    "--cool",
    "cool_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Fragment-level contact map written by instagraal-pre (in the coordinates of the initial contigs).",
)
@click.option(
    "-f",
    "--fasta",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Initial assembly FASTA (the one given to instagraal-pre).",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    default="refined",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Output directory (created if missing).",
)
@click.option("-b", "--binsize", default=DEFAULT_BINSIZE, show_default=True, type=int, help="Bin size (bp) for contig contacts.")
@click.option(
    "-w",
    "--window",
    default=DEFAULT_WINDOW,
    show_default=True,
    type=int,
    help="Length (bp) next to each contig end whose contacts are compared.",
)
@click.option(
    "-l",
    "--min-contig-length",
    default=DEFAULT_MIN_CONTIG_LENGTH,
    show_default=True,
    type=int,
    help="Shorter contigs are not joined and are output as their own scaffolds.",
)
@click.option(
    "--min-score",
    default=DEFAULT_MIN_SCORE,
    show_default=True,
    type=float,
    help="Minimum log-likelihood ratio (junction vs background) for a join.",
)
@click.option(
    "--min-margin",
    default=DEFAULT_MIN_MARGIN,
    show_default=True,
    type=float,
    help="Minimum lead of a join over each end's next best candidate (higher: fewer, more reliable joins).",
)
@click.option(
    "-j",
    "--junction",
    default="",
    help="Junction sequence inserted between contigs in refined_genome.fa (e.g. NNNNNN).",
)
@click.option(
    "--keep-unresolved",
    is_flag=True,
    default=False,
    help="Also keep joins of the input layout whose ends found no confident partner, if they reach --min-score.",
)
@click.option(
    "--qc-only",
    is_flag=True,
    default=False,
    help="Only score the input layout's joins (junctions.tsv, refine_report.txt); do not re-scaffold.",
)
def main(
    info_frags: pathlib.Path,
    cool_path: pathlib.Path,
    fasta: pathlib.Path,
    output_dir: pathlib.Path,
    binsize: int,
    window: int,
    min_contig_length: int,
    min_score: float,
    min_margin: float,
    junction: str,
    keep_unresolved: bool,
    qc_only: bool,
) -> None:
    """Check and fix the contig order of an instaGRAAL assembly from local Hi-C evidence.

    instaGRAAL's scaffolder can fold chromosomes whose two replication arms are
    aligned (an X in the contact map), interleaving the arms. This command scores
    every contig junction from the contacts next to the two contig ends only, flags
    the input layout's unsupported or contradicted joins, and re-joins whole contigs
    whose ends are each other's clear best partner.

    \b
    Outputs (in OUTPUT_DIR):
      refined_info_frags.txt  new layout (whole contigs), usable by instagraal-post
      refined_genome.fa       refined assembly
      junctions.tsv           score of every input and refined join
      refine_report.txt       summary, incl. GC skew of large scaffolds
    """
    if window < binsize:
        raise click.BadParameter("--window must be at least --binsize.", param_hint="--window")
    try:
        run_refine(
            info_frags=info_frags,
            cool_path=cool_path,
            fasta=fasta,
            output_dir=output_dir,
            binsize=binsize,
            window=window,
            min_contig_length=min_contig_length,
            min_score=min_score,
            min_margin=min_margin,
            junction=junction,
            keep_unresolved=keep_unresolved,
            qc_only=qc_only,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
