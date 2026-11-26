"""Click CLI for the ``instagraal-post`` post-processing command."""

import json
import pathlib

import click

from ..parse_info_frags import DEFAULT_JUNCTION_SEQUENCE
from ..post import run_post


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "pairs",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
)
@click.argument(
    "new_info_frags",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    metavar="NEW_INFO_FRAGS",
)
@click.option(
    "-r",
    "--resolutions",
    required=True,
    help=(
        "Resolution(s) of the output .mcool, in bp, comma-separated (e.g. '1000,5000,10000,25000,100000'). "
        "The smallest is the base resolution, at which the lifted pairs are binned; the others are "
        "aggregated from it, so each must be a multiple of a smaller one. A single value gives one resolution."
    ),
)
@click.option(
    "-o",
    "--output-dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Directory where output files will be written.",
)
@click.option(
    "--cool-name",
    default=None,
    help="Base name for the output .cool/.mcool files (default: pairs file stem).",
)
@click.option(
    "--junction",
    default=len(DEFAULT_JUNCTION_SEQUENCE),
    show_default=True,
    type=int,
    help=(
        "Length (bp) of the junction sequence inserted between fragments "
        "from different source contigs during polishing.  "
        "Must be the length of the --junction sequence used with instagraal-polish "
        "(0 for --junction '').  The default matches instagraal-polish's default junction (NNNNNN)."
    ),
)
@click.option(
    "--balance/--no-balance",
    default=True,
    show_default=True,
    help=("Apply ICE balancing at each zoom level.  Enabled by default; use --no-balance to skip."),
)
@click.option(
    "--balance-args",
    default=None,
    help=(
        "Extra arguments for ICE balancing as a JSON object, e.g. "
        '\'{"max_iters": 2000, "mad_max": 10}\'.  '
        "Keys correspond to cooler.balance_cooler keyword arguments."
    ),
)
@click.option(
    "--fasta",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help=(
        "Original reference FASTA (plain or .gz).  Currently unused — "
        "all coordinate information is derived from NEW_INFO_FRAGS.  "
        "Reserved for future GC-content annotation of output bins."
    ),
)
def main(
    pairs: pathlib.Path,
    new_info_frags: pathlib.Path,
    resolutions: str,
    output_dir: pathlib.Path,
    cool_name: str | None,
    junction: int,
    balance: bool,
    balance_args: str | None,
    fasta: pathlib.Path | None,
) -> None:
    """Post-process instaGRAAL results into a multi-resolution contact map.

    Lifts over the original Hi-C read pairs from the old reference genome to
    the new assembly produced by instagraal-polish or instagraal-refine, bins
    them, and writes an .mcool at the requested resolution(s), with .cool files
    of the scaffolds and of the contigs.

    \b
    Typical workflow:
      1. instagraal-pre     genome.fa  reads.pairs  -e DpnII -o hic/
      2. instagraal         hic/  genome.fa  -o out/
      3. instagraal-polish  -i out/hic/test_mcmc_4/info_frags.txt -f genome.fa -o polished/
      4. instagraal-post    reads.pairs  polished/new_info_frags.txt  \\
                            -r 1000,5000,25000,100000  -o post/

    \b
    PAIRS           Hi-C pairs file aligned to the *original* reference
                    (plain or .gz, 4DN format).
    NEW_INFO_FRAGS  Layout of the new assembly: new_info_frags.txt of
                    instagraal-polish, refined_info_frags.txt of
                    instagraal-refine, or final_info_frags.txt of
                    instagraal-endtoend.
    """
    parsed_balance_args: dict | None = None
    if balance_args is not None:
        try:
            parsed_balance_args = json.loads(balance_args)
        except json.JSONDecodeError as exc:
            raise click.BadParameter(f"--balance-args must be valid JSON: {exc}") from exc

    run_post(
        pairs=pairs,
        new_info_frags=new_info_frags,
        output_dir=output_dir,
        resolutions=resolutions,
        cool_name=cool_name,
        junction_len=junction,
        balance=balance,
        balance_args=parsed_balance_args,
    )
