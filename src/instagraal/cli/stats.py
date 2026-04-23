"""Click CLI for the ``instagraal-stats`` assembly statistics command."""

import pathlib

import click

from ..assembly_stats import (
    compute_assembly_stats,
    format_assembly_stats,
    format_comparison_table,
)
from ..version import __version__ as VERSION_NUMBER


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION_NUMBER, "-V", "--version")
@click.argument(
    "fasta_files",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    metavar="ASSEMBLY.FA [ASSEMBLY.FA ...]",
)
@click.option(
    "-l",
    "--labels",
    default=None,
    help="Comma-separated labels to use as column headers (one per file). Defaults to the file basenames.",
)
def main(
    fasta_files: tuple[pathlib.Path, ...],
    labels: str | None,
) -> None:
    """Compute and display assembly statistics for one or more FASTA files.

    When a single file is provided the output is a single-column summary
    table.  When multiple files are provided their statistics are printed
    side-by-side for easy comparison.

    \b
    Metrics reported (no external databases required):
      Sequences        Number of sequences (contigs / scaffolds)
      Total length     Sum of all sequence lengths in bp
      Largest          Length of the longest sequence
      Shortest         Length of the shortest sequence
      Mean / Median    Arithmetic mean and median sequence length
      N50 / L50        Standard N50 and L50 contig statistics
      N90 / L90        Standard N90 and L90 contig statistics
      GC content       Fraction of G+C bases across the whole assembly
    """
    if labels is not None:
        label_list = [lb.strip() for lb in labels.split(",")]
        if len(label_list) != len(fasta_files):
            raise click.UsageError(
                f"--labels supplied {len(label_list)} label(s) but {len(fasta_files)} file(s) were given."
            )
    else:
        label_list = [f.name for f in fasta_files]

    results = {}
    for path, label in zip(fasta_files, label_list):
        results[label] = compute_assembly_stats(str(path))

    if len(fasta_files) == 1:
        label = label_list[0]
        click.echo(format_assembly_stats(results[label], label=label))
    else:
        click.echo(format_comparison_table(results))
