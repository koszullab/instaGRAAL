"""Click CLI for the ``instagraal-pre`` pre-processing command."""

import pathlib

import click

from ..pre import run_pre


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "fasta",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "pairs",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.option(
    "--enzyme",
    "-e",
    required=True,
    help="Restriction enzyme name(s), comma-separated (e.g. DpnII or DpnII,HinfI).",
)
@click.option(
    "--output-dir",
    "-o",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Directory where output files will be written.",
)
@click.option(
    "--cool-name",
    default=None,
    help="Base name for the output .cool file (default: pairs file stem).",
)
def main(
    fasta: pathlib.Path,
    pairs: pathlib.Path,
    enzyme: str,
    output_dir: pathlib.Path,
    cool_name: str | None,
) -> None:
    """Pre-process Hi-C data for instaGRAAL.

    Digests FASTA with the given restriction enzyme(s), bins the read pairs
    from PAIRS into restriction fragments, and writes:

    \b
      fragments_list.txt
      info_contigs.txt
      abs_fragments_contacts_weighted.txt
      <name>.cool

    to OUTPUT_DIR (which is also a valid instaGRAAL input folder).

    FASTA   Genome assembly FASTA file (plain or .gz).

    PAIRS   Hi-C pairs file in 4DN pairs format (plain or .gz).
            Required columns: readID chr1 pos1 chr2 pos2 strand1 strand2.
    """
    enzymes = [e.strip() for e in enzyme.split(",") if e.strip()]
    run_pre(fasta, pairs, enzymes, output_dir, cool_name)
