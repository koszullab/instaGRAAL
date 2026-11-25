"""Click CLI for the ``instagraal-refine`` command."""

import pathlib

import click

from ..refine import (
    DEFAULT_BINSIZE,
    DEFAULT_MIN_CONTIG_LENGTH,
    DEFAULT_MIN_MARGIN,
    DEFAULT_MIN_SCORE,
    DEFAULT_SHORT_MARGIN,
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
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help=(
        "Layout to check: new_info_frags.txt from instagraal-polish (or instagraal's info_frags.txt). Optional: the "
        "refined assembly depends on the contact map and the contigs only; the layout is scored for QC."
    ),
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
    help="Shorter contigs are not joined to each other; see --place-short.",
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
    "--place-short/--no-place-short",
    default=True,
    show_default=True,
    help=(
        "Place each contig shorter than --min-contig-length, whole, between the two contigs (or at the scaffold end) "
        "its contacts fit best; otherwise output it as its own scaffold."
    ),
)
@click.option(
    "--short-margin",
    default=DEFAULT_SHORT_MARGIN,
    show_default=True,
    type=float,
    help="Minimum lead of a short contig's best place over its next best (higher: fewer, more reliable placements).",
)
@click.option(
    "--recenter/--no-recenter",
    default=True,
    show_default=True,
    help=(
        "Cut each circular scaffold of at least 500 kb at the replication terminus found from its GC skew, so that "
        "the origin sits near its middle (left as is if the skew does not show two clear replichores)."
    ),
)
@click.option(
    "--origin-first",
    is_flag=True,
    default=False,
    help="With --recenter, start circular scaffolds at the origin instead.",
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
    info_frags: pathlib.Path | None,
    cool_path: pathlib.Path,
    fasta: pathlib.Path,
    output_dir: pathlib.Path,
    binsize: int,
    window: int,
    min_contig_length: int,
    min_score: float,
    min_margin: float,
    junction: str,
    place_short: bool,
    short_margin: float,
    recenter: bool,
    origin_first: bool,
    keep_unresolved: bool,
    qc_only: bool,
) -> None:
    """Check and fix the contig order of an instaGRAAL assembly from local Hi-C evidence.

    instaGRAAL's scaffolder can fold chromosomes whose two replication arms are
    aligned (an X in the contact map), interleaving the arms. This command scores
    every contig junction from the contacts next to the two contig ends only, flags
    the input layout's unsupported or contradicted joins, and re-joins whole contigs
    whose ends are each other's clear best partner, then places the contigs too short
    to be joined where their contacts clearly fit, and recenters circular scaffolds
    on their GC skew. The result does not depend on the input layout, so the GPU
    scaffolding step can be skipped: run it on instagraal-pre's output alone.

    \b
    Outputs (in OUTPUT_DIR):
      refined_info_frags.txt  new layout (whole contigs), usable by instagraal-post
      refined_genome.fa       refined assembly
      junctions.tsv           score of every input and refined join
      refine_report.txt       summary, incl. GC skew of large scaffolds
    """
    if window < binsize:
        raise click.BadParameter("--window must be at least --binsize.", param_hint="--window")
    if info_frags is None and (qc_only or keep_unresolved):
        raise click.UsageError("--qc-only and --keep-unresolved need an input layout (--info-frags).")
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
            place_short=place_short,
            short_margin=short_margin,
            recenter=recenter,
            origin_first=origin_first,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
