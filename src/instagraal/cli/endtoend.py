"""Click CLI for the ``instagraal-endtoend`` full pipeline command."""

import contextlib
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

import click

from ..version import __version__ as VERSION_NUMBER

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LEVEL = 4
DEFAULT_CYCLES = 100
DEFAULT_RESOLUTIONS = "1000"


# ---------------------------------------------------------------------------
# Helpers (shared logic with instagraal-test)
# ---------------------------------------------------------------------------


def _check_nvcc() -> None:
    """Verify that nvcc (NVIDIA CUDA Compiler) is available in PATH."""
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        version_line = result.stdout.strip().split("\n")[-1]
        click.echo(f"  nvcc: {version_line}")
    except FileNotFoundError:
        raise click.ClickException(
            "nvcc not found in PATH.  CUDA Toolkit SDK must be installed and nvcc must be accessible."
        ) from None
    except subprocess.TimeoutExpired:
        raise click.ClickException("nvcc check timed out.") from None
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"nvcc check failed: {exc.stderr}") from exc


def _check_gpu(device: int) -> None:
    """Verify that pycuda can initialise the requested CUDA device."""
    try:
        import pycuda
    except ImportError as exc:
        raise click.ClickException("pycuda is not installed.  instagraal requires it along with a CUDA-capable GPU.") from exc
    click.echo(f"  pycuda version: {pycuda.VERSION_TEXT}")

    try:
        import pycuda.driver as cuda
    except ImportError as exc:
        raise click.ClickException(
            "pycuda cannot import the driver module.  Check your pycuda installation and CUDA setup."
        ) from exc
    version = cuda.get_version()
    click.echo(f"  Detected CUDA runtime version {version[0]}.{version[1]}.{version[2]}.")

    try:
        cuda.init()
    except Exception as exc:
        raise click.ClickException(f"CUDA initialisation failed: {exc}") from exc
    click.echo("  Cuda successfully initialised")

    n_devices = cuda.Device.count()
    if n_devices == 0:
        raise click.ClickException("No CUDA devices found.  instagraal requires at least one GPU.")
    if device >= n_devices:
        raise click.ClickException(f"Requested device {device} but only {n_devices} device(s) available (0-{n_devices - 1}).")

    dev = cuda.Device(device)
    click.echo(f"  GPU {device}: {dev.name()} ({dev.total_memory() // 1024**2} MB)")
    _check_nvcc()


def _cmd_echo(cmd: list) -> None:
    """Print a command in shell-quoted form before running it."""
    click.echo("  $ " + " ".join(shlex.quote(str(a)) for a in cmd))


def _run_cmd(cmd: list, _record: list | None = None, dry_run: bool = False) -> None:
    """Print then optionally execute a CLI command, streaming output to the console.

    When *dry_run* is ``True`` the command is printed but not executed.
    """
    _cmd_echo(cmd)
    if _record is not None:
        _record.append(" ".join(shlex.quote(str(a)) for a in cmd))
    if not dry_run:
        subprocess.run([str(a) for a in cmd], check=True)


@contextlib.contextmanager
def _step(label: str):
    """Context manager that prints a step header and elapsed time on exit."""
    click.echo(f"\n{label}")
    t0 = time.monotonic()
    yield
    click.echo(f"  Done. ({time.monotonic() - t0:.1f}s)")


# ---------------------------------------------------------------------------
# Internal pipeline runner
# ---------------------------------------------------------------------------


def _run_endtoend(
    fasta: pathlib.Path,
    pairs: pathlib.Path,
    enzymes: list[str],
    output_dir: pathlib.Path,
    device: int,
    level: int,
    cycles: int,
    resolutions: str,
    bomb: bool,
    circular: bool,
    save_matrix: bool,
    save_pickle: bool,
    simple: bool,
    quiet: bool,
    debug: bool,
    balance: bool,
    balance_args: str | None,
    coverage_std: float,
    neighborhood: int,
    min_scaffold_size: int,
    min_scaffold_length: int,
    junction: str,
    criterion: str | None,
    cool_name: str | None,
    dry_run: bool = False,
) -> list[str]:
    """Execute the full instaGRAAL pipeline on user-supplied data.

    Returns a list of shell-quoted commands that were executed (or would be
    executed when *dry_run* is ``True``), for the recap.
    """
    ran_cmds: list[str] = []

    bin_dir = pathlib.Path(sys.executable).parent

    def cmd(name: str) -> pathlib.Path:
        found = shutil.which(name)
        if found:
            return pathlib.Path(found)
        return bin_dir / name

    # -- 1. GPU check --------------------------------------------------------
    with _step("[1/6] Checking GPU …"):
        if not dry_run:
            _check_gpu(device)
        else:
            click.echo("  (skipped — dry run)")

    # -- 2. instagraal-pre ---------------------------------------------------
    with _step("[2/6] Running instagraal-pre …"):
        pre_dir = output_dir / "pre"
        pre_dir.mkdir(exist_ok=True)
        pre_cmd = [
            cmd("instagraal-pre"),
            fasta,
            pairs,
            "--enzyme",
            ",".join(enzymes),
            "--output-dir",
            pre_dir,
        ]
        if cool_name is not None:
            pre_cmd += ["--cool-name", cool_name]
        _run_cmd(pre_cmd, _record=ran_cmds, dry_run=dry_run)

    # -- 3. instagraal (MCMC scaffolding) ------------------------------------
    with _step("[3/6] Running instagraal …"):
        mcmc_base_dir = output_dir / "mcmc"
        mcmc_base_dir.mkdir(exist_ok=True)
        mcmc_cmd = [
            cmd("instagraal"),
            pre_dir,
            fasta,
            "--output-dir",
            mcmc_base_dir,
            "--level",
            level,
            "--cycles",
            cycles,
            "--device",
            device,
            "--coverage-std",
            coverage_std,
            "--neighborhood",
            neighborhood,
        ]
        if bomb:
            mcmc_cmd.append("--bomb")
        if circular:
            mcmc_cmd.append("--circular")
        if save_matrix:
            mcmc_cmd.append("--save-matrix")
        if save_pickle:
            mcmc_cmd.append("--save-pickle")
        if simple:
            mcmc_cmd.append("--simple")
        if quiet:
            mcmc_cmd.append("--quiet")
        if debug:
            mcmc_cmd.append("--debug")
        _run_cmd(mcmc_cmd, _record=ran_cmds, dry_run=dry_run)

    mcmc_out = mcmc_base_dir / pre_dir.name / f"test_mcmc_{level}"
    info_frags = mcmc_out / "info_frags.txt"
    if not dry_run and not info_frags.exists():
        raise click.ClickException(f"instagraal did not produce info_frags.txt at expected path:\n  {info_frags}")

    # -- 4. instagraal-polish ------------------------------------------------
    with _step("[4/6] Running instagraal-polish …"):
        polish_dir = output_dir / "polished"
        polish_dir.mkdir(exist_ok=True)
        polish_cmd = [
            cmd("instagraal-polish"),
            "--input",
            info_frags,
            "--fasta",
            fasta,
            "--output-dir",
            polish_dir,
            "--min-scaffold-size",
            min_scaffold_size,
            "--min-scaffold-length",
            min_scaffold_length,
        ]
        if junction:
            polish_cmd += ["--junction", junction]
        if criterion is not None:
            polish_cmd += ["--criterion", criterion]
        _run_cmd(polish_cmd, _record=ran_cmds, dry_run=dry_run)
    polished_fa = polish_dir / "polished_genome.fa"
    new_info_frags = polish_dir / "new_info_frags.txt"

    # -- 5. instagraal-post --------------------------------------------------
    with _step("[5/6] Running instagraal-post …"):
        post_dir = output_dir / "post"
        post_dir.mkdir(exist_ok=True)
        post_cmd = [
            cmd("instagraal-post"),
            pairs,
            new_info_frags,
            "--resolutions",
            resolutions,
            "--output-dir",
            post_dir,
            "--junction",
            len(junction) if junction else 6,
        ]
        if not balance:
            post_cmd.append("--no-balance")
        if balance_args is not None:
            post_cmd += ["--balance-args", balance_args]
        if cool_name is not None:
            post_cmd += ["--cool-name", cool_name]
        _run_cmd(post_cmd, _record=ran_cmds, dry_run=dry_run)

    # -- 6. instagraal-stats -------------------------------------------------
    with _step("[6/6] Running instagraal-stats …"):
        _run_cmd(
            [
                cmd("instagraal-stats"),
                fasta,
                polished_fa,
                "--labels",
                "Input assembly,Polished assembly",
            ],
            _record=ran_cmds,
            dry_run=dry_run,
        )

    return ran_cmds


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION_NUMBER, "-V", "--version")
@click.argument(
    "fasta",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    metavar="FASTA",
)
@click.argument(
    "pairs",
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    metavar="PAIRS",
)
@click.option(
    "-e",
    "--enzyme",
    required=True,
    help="Restriction enzyme name(s), comma-separated (e.g. DpnII or DpnII,HinfI).",
)
@click.option(
    "-o",
    "--output-dir",
    "output_dir",
    default="out",
    show_default=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Root directory for all pipeline outputs (created if missing).",
)
@click.option(
    "--device",
    default=0,
    show_default=True,
    type=int,
    help="CUDA device index to use.",
)
@click.option(
    "-l",
    "--level",
    default=DEFAULT_LEVEL,
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
    default=DEFAULT_CYCLES,
    show_default=True,
    type=int,
    help=("Number of MCMC iterations per bin. A high number has diminishing returns but a minimum is needed for convergence."),
)
@click.option(
    "-r",
    "--resolutions",
    default=DEFAULT_RESOLUTIONS,
    show_default=True,
    help=(
        "Resolution(s) for the output .mcool passed to instagraal-post, "
        "as a comma-separated list of base-pair values "
        "(e.g. '1000,5000,10000') or a single integer N for a geometric series."
    ),
)
@click.option(
    "-c",
    "--coverage-std",
    default=1.0,
    show_default=True,
    type=float,
    help=("Number of standard deviations below the mean coverage below which fragments are filtered out prior to binning."),
)
@click.option(
    "-N",
    "--neighborhood",
    default=5,
    show_default=True,
    type=int,
    help="Number of neighbors sampled for potential mutations per bin.",
)
@click.option(
    "-b",
    "--bomb/--no-bomb",
    default=True,
    show_default=True,
    help="Explode the genome prior to scaffolding.",
)
@click.option(
    "-C",
    "--circular",
    is_flag=True,
    default=False,
    help="Indicate that the genome is circular.",
)
@click.option(
    "--save-matrix/--no-save-matrix",
    default=True,
    show_default=True,
    help="Save a preview of the contact map after each MCMC cycle.",
)
@click.option(
    "--balance/--no-balance",
    default=True,
    show_default=True,
    help="Apply ICE balancing at each zoom level in instagraal-post.",
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
    "-s",
    "--min-scaffold-size",
    default=0,
    show_default=True,
    type=int,
    help="Minimum scaffold size in bins passed to instagraal-polish.",
)
@click.option(
    "-L",
    "--min-scaffold-length",
    default=0,
    show_default=True,
    type=int,
    help="Minimum scaffold length in bp passed to instagraal-polish.",
)
@click.option(
    "-j",
    "--junction",
    default="NNNNNN",
    show_default=True,
    help=(
        "Junction sequence inserted between stitched bins by instagraal-polish. "
        "Its length is automatically forwarded to instagraal-post (--junction). "
    ),
)
@click.option(
    "--criterion",
    default=None,
    help="Block criterion stringency for intragraal-polish inversion/inversion2 modes.",
)
@click.option(
    "--cool-name",
    default=None,
    help=("Base name for the output .cool/.mcool files. Forwarded to both instagraal-pre and instagraal-post."),
)
@click.option(
    "--simple",
    is_flag=True,
    default=False,
    help="Only perform operations at the edge of the contigs (instagraal).",
)
@click.option(
    "--save-pickle",
    is_flag=True,
    default=False,
    help="Dump all instagraal run info into a pickle file.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Only display warnings and errors from instagraal.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Display debug information from instagraal. Overrides --quiet.",
)
@click.option(
    "--dry-run",
    "dry_run",
    is_flag=True,
    default=False,
    help=(
        "Print the commands that would be executed without running them. "
        "Useful for inspecting the full pipeline before committing to a long run."
    ),
)
def main(
    fasta: pathlib.Path,
    pairs: pathlib.Path,
    enzyme: str,
    output_dir: pathlib.Path,
    device: int,
    level: int,
    cycles: int,
    resolutions: str,
    coverage_std: float,
    neighborhood: int,
    bomb: bool,
    circular: bool,
    save_matrix: bool,
    save_pickle: bool,
    simple: bool,
    quiet: bool,
    debug: bool,
    dry_run: bool,
    balance: bool,
    balance_args: str | None,
    min_scaffold_size: int,
    min_scaffold_length: int,
    junction: str,
    criterion: str | None,
    cool_name: str | None,
) -> None:
    """Run the complete instaGRAAL pipeline end-to-end.

    \b
    Steps
    -----
      1. Check that a CUDA-capable GPU is accessible.
      2. instagraal-pre    — digest FASTA and bin Hi-C pairs into fragments.
      3. instagraal        — MCMC-based scaffolding.
      4. instagraal-polish — full polishing pipeline.
      5. instagraal-post   — remap pairs to new assembly, write .mcool.
      6. instagraal-stats  — compare input vs. polished assembly statistics.

    \b
    Required inputs
    ---------------
      FASTA   Genome assembly FASTA file (plain or .gz).
      PAIRS   Hi-C pairs file in 4DN pairs format (plain or .gz).

    All intermediate and final results are written under OUTPUT_DIR.
    """
    enzymes = [e.strip() for e in enzyme.split(",") if e.strip()]
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Output directory : {output_dir.resolve()}")
    click.echo(f"FASTA            : {fasta.resolve()}")
    click.echo(f"Pairs            : {pairs.resolve()}")
    click.echo(f"Enzyme(s)        : {', '.join(enzymes)}")
    if dry_run:
        click.echo("(dry run — commands will be printed but not executed)")

    _start = time.monotonic()
    ran_cmds = _run_endtoend(
        fasta=fasta,
        pairs=pairs,
        enzymes=enzymes,
        output_dir=output_dir,
        device=device,
        level=level,
        cycles=cycles,
        resolutions=resolutions,
        bomb=bomb,
        circular=circular,
        save_matrix=save_matrix,
        save_pickle=save_pickle,
        simple=simple,
        quiet=quiet,
        debug=debug,
        balance=balance,
        balance_args=balance_args,
        coverage_std=coverage_std,
        neighborhood=neighborhood,
        min_scaffold_size=min_scaffold_size,
        min_scaffold_length=min_scaffold_length,
        junction=junction,
        criterion=criterion,
        cool_name=cool_name,
        dry_run=dry_run,
    )
    _elapsed = time.monotonic() - _start
    if dry_run:
        click.echo(f"\n[instagraal-endtoend] DRY RUN completed  (total time: {_elapsed:.1f}s).")
        click.echo("\nCommands that would be executed:")
    else:
        click.echo(f"\n[instagraal-endtoend] ALL STEPS COMPLETED  (total time: {_elapsed:.1f}s).")
        click.echo("\nCommands executed (for reproducibility):")
    click.echo("\n```")
    for c in ran_cmds:
        click.echo(c)
    click.echo("```")
