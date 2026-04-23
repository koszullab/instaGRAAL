"""Click CLI for the ``instagraal-polish`` assembly polishing command."""

import pathlib

import click

from ..parse_info_frags import (
    DEFAULT_CRITERION,
    DEFAULT_CRITERION_2,
    DEFAULT_MIN_SCAFFOLD_SIZE,
    DEFAULT_NEW_INFO_FRAGS_NAME,
    correct_spurious_inversions,
    find_lost_dna,
    integrate_lost_dna,
    parse_info_frags,
    plot_info_frags,
    rearrange_intra_scaffolds,
    reorient_consecutive_blocks,
    remove_spurious_insertions,
    write_fasta,
    write_info_frags,
)

VALID_MODES = (
    "fasta",
    "singleton",
    "inversion",
    "inversion2",
    "rearrange",
    "reincorporation",
    "polishing",
    "plot",
)

POLISHED_GENOME_NAME = "polished_genome.fa"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--mode",
    required=True,
    type=click.Choice(VALID_MODES, case_sensitive=False),
    help=(
        "\b\nProcessing mode:\n"
        "  fasta           Write a new genome FASTA from info_frags + reference.\n"
        "  singleton       Remove spurious singleton insertions.\n"
        "  inversion       Correct spurious inversions (colinear criterion).\n"
        "  inversion2      Correct spurious inversions (blocks criterion).\n"
        "  rearrange       Rearrange intra-scaffold blocks.\n"
        "  reincorporation Reincorporate lost DNA from reference.\n"
        "  polishing       Full pipeline: rearrange + inversion2 + reincorporation + fasta.\n"
        "  plot            Plot a visual summary of the scaffolds."
    ),
)
@click.option(
    "-i",
    "--input",
    "info_frags",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Input info_frags.txt file to process.",
)
@click.option(
    "-f",
    "--fasta",
    "init_fasta",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Reference FASTA file (required for fasta, reincorporation, and polishing modes).",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Output directory (created if missing). All output files are written here.",
)
@click.option(
    "-c",
    "--criterion",
    default=None,
    help="Block criterion stringency (used for inversion/inversion2 modes).",
)
@click.option(
    "-s",
    "--min-scaffold-size",
    default=DEFAULT_MIN_SCAFFOLD_SIZE,
    show_default=True,
    type=int,
    help="Minimum scaffold size in bins.",
)
@click.option(
    "-j",
    "--junction",
    default="",
    help="Junction sequence inserted between stitched bins.",
)
def main(
    mode: str,
    info_frags: pathlib.Path,
    init_fasta: pathlib.Path | None,
    output_dir: pathlib.Path,
    criterion: str | None,
    min_scaffold_size: int,
    junction: str,
) -> None:
    """Polish and post-process instaGRAAL assemblies."""
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaffolds = {
        name: scaffold
        for name, scaffold in parse_info_frags(str(info_frags)).items()
        if len(scaffold) > min_scaffold_size
    }

    if mode == "fasta":
        if init_fasta is None:
            raise click.UsageError("A reference FASTA file must be provided (--fasta) for 'fasta' mode.")
        write_fasta(
            init_fasta=str(init_fasta),
            info_frags=str(info_frags),
            junction=junction,
            output=str(output_dir / POLISHED_GENOME_NAME),
        )

    elif "singleton" in mode:
        new_scaffolds = remove_spurious_insertions(scaffolds)
        write_info_frags(new_scaffolds, output=str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME))

    elif mode == "inversion":
        effective_criterion = criterion or DEFAULT_CRITERION
        new_scaffolds = correct_spurious_inversions(scaffolds=scaffolds, criterion=effective_criterion)
        write_info_frags(new_scaffolds, output=str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME))

    elif mode == "inversion2":
        effective_criterion = criterion or DEFAULT_CRITERION_2
        new_scaffolds = reorient_consecutive_blocks(scaffolds=scaffolds, mode=effective_criterion)
        write_info_frags(new_scaffolds, output=str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME))

    elif "rearrange" in mode:
        new_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        write_info_frags(new_scaffolds, output=str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME))

    elif "reincorporation" in mode:
        if init_fasta is None:
            raise click.UsageError("A reference FASTA file must be provided (--fasta) for 'reincorporation' mode.")
        removed = find_lost_dna(init_fasta=str(init_fasta), scaffolds=scaffolds)
        new_scaffolds = integrate_lost_dna(scaffolds=scaffolds, lost_dna_positions=removed)
        write_info_frags(new_scaffolds, output=str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME))

    elif "polishing" in mode:
        if init_fasta is None:
            raise click.UsageError("A reference FASTA file must be provided (--fasta) for 'polishing' mode.")
        info_frags_file = str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME)
        genome_file = str(output_dir / POLISHED_GENOME_NAME)
        arranged_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        reoriented_scaffolds = reorient_consecutive_blocks(arranged_scaffolds)
        removed = find_lost_dna(init_fasta=str(init_fasta), scaffolds=reoriented_scaffolds)
        new_scaffolds = integrate_lost_dna(scaffolds=reoriented_scaffolds, lost_dna_positions=removed)
        write_info_frags(new_scaffolds, output=info_frags_file)
        write_fasta(
            init_fasta=str(init_fasta),
            info_frags=info_frags_file,
            output=genome_file,
            junction=junction,
        )

    elif mode == "plot":
        plot_info_frags(scaffolds)
