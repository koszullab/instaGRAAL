"""Click CLI for the ``instagraal-polish`` assembly polishing command."""

import pathlib

import click

from ..assembly_stats import print_assembly_stats
from ..parse_info_frags import (
    DEFAULT_CRITERION,
    DEFAULT_CRITERION_2,
    DEFAULT_MIN_SCAFFOLD_SIZE,
    DEFAULT_NEW_INFO_FRAGS_NAME,
    correct_spurious_inversions,
    find_lost_dna,
    plot_contig_composition,
    integrate_lost_dna,
    parse_info_frags,
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
)

DEFAULT_MIN_SCAFFOLD_LENGTH = 0  # in bp
POLISHED_GENOME_NAME = "polished_genome.fa"


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--mode",
    default=None,
    type=click.Choice(VALID_MODES, case_sensitive=False),
    help=(
        "\b\nProcessing mode (default: run full polishing pipeline):\n"
        "  rearrange       Rearrange intra-scaffold blocks.\n"
        "  inversion2      Correct spurious inversions (blocks criterion).\n"
        "  reincorporation Reincorporate lost DNA from reference.\n"
        "  fasta           Write a new genome FASTA from info_frags + reference.\n"
        "  singleton       Remove spurious singleton insertions.\n"
        "  inversion       Correct spurious inversions (colinear criterion).\n"
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
    help=(
        "The initial reference FASTA file, before running instaGRAAL. "
        "(required for 'fasta', 'reincorporation', and 'polishing' modes)"
    ),
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    default="out",
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
    "-l",
    "--min-scaffold-length",
    default=DEFAULT_MIN_SCAFFOLD_LENGTH,
    show_default=True,
    type=int,
    help="Minimum scaffold length in bp.",
)
@click.option(
    "-j",
    "--junction",
    default="",
    help="Junction sequence inserted between stitched bins.",
)
def main(
    mode: str | None,
    info_frags: pathlib.Path,
    init_fasta: pathlib.Path | None,
    output_dir: pathlib.Path,
    criterion: str | None,
    min_scaffold_size: int,
    min_scaffold_length: int,
    junction: str,
) -> None:
    """Polish and post-process instaGRAAL assemblies.

    By default (no --mode given), runs the full polishing pipeline:
    rearrange → inversion2 → reincorporation → fasta.
    A reference FASTA (--fasta) is required for this default pipeline.
    """
    mode = mode or "polishing"
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scaffolds = {
        name: scaffold for name, scaffold in parse_info_frags(str(info_frags)).items() if len(scaffold) > min_scaffold_size
    }
    print(len(scaffolds), "scaffolds retained after filtering by minimum number of bins [", min_scaffold_size, "].")
    scaffolds = {
        name: scaffold
        for name, scaffold in scaffolds.items()
        if sum(end - start for _, _, start, end, _ in scaffold) >= min_scaffold_length
    }
    print(len(scaffolds), "scaffolds retained after filtering by minimum length [", min_scaffold_length, "].")

    if mode == "fasta":
        if init_fasta is None:
            raise click.UsageError("A reference FASTA file must be provided (--fasta) for 'fasta' mode.")
        genome_file = str(output_dir / POLISHED_GENOME_NAME)
        write_fasta(
            init_fasta=str(init_fasta),
            info_frags=str(info_frags),
            junction=junction,
            output=genome_file,
        )
        print_assembly_stats(genome_file, label="Assembly (fasta mode)")

    elif "singleton" in mode:
        new_scaffolds = remove_spurious_insertions(scaffolds)
        info_frags_file = str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME)
        write_info_frags(new_scaffolds, output=info_frags_file)
        plot_contig_composition(info_frags_file, output_path=output_dir / "contig_composition.png")

    elif mode == "inversion":
        effective_criterion = criterion or DEFAULT_CRITERION
        new_scaffolds = correct_spurious_inversions(scaffolds=scaffolds, criterion=effective_criterion)
        info_frags_file = str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME)
        write_info_frags(new_scaffolds, output=info_frags_file)
        plot_contig_composition(info_frags_file, output_path=output_dir / "contig_composition.png")

    elif mode == "inversion2":
        effective_criterion = criterion or DEFAULT_CRITERION_2
        new_scaffolds = reorient_consecutive_blocks(scaffolds=scaffolds, mode=effective_criterion)
        info_frags_file = str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME)
        write_info_frags(new_scaffolds, output=info_frags_file)
        plot_contig_composition(info_frags_file, output_path=output_dir / "contig_composition.png")

    elif "rearrange" in mode:
        new_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        info_frags_file = str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME)
        write_info_frags(new_scaffolds, output=info_frags_file)
        plot_contig_composition(info_frags_file, output_path=output_dir / "contig_composition.png")

    elif "reincorporation" in mode:
        if init_fasta is None:
            raise click.UsageError("A reference FASTA file must be provided (--fasta) for 'reincorporation' mode.")
        removed = find_lost_dna(init_fasta=str(init_fasta), scaffolds=scaffolds)
        new_scaffolds = integrate_lost_dna(scaffolds=scaffolds, lost_dna_positions=removed)
        info_frags_file = str(output_dir / DEFAULT_NEW_INFO_FRAGS_NAME)
        write_info_frags(new_scaffolds, output=info_frags_file)
        plot_contig_composition(info_frags_file, output_path=output_dir / "contig_composition.png")

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
        plot_contig_composition(info_frags_file, output_path=output_dir / "contig_composition.png")
        print_assembly_stats(genome_file, label="Assembly (polishing mode)")
