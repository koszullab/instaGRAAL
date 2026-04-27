"""Click CLI for the ``instagraal-test`` end-to-end functional test command."""

import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import click

from ..version import __version__ as VERSION_NUMBER

# ---------------------------------------------------------------------------
# Zenodo configuration
# Update ZENODO_RECORD_ID once the dataset is published.
# ---------------------------------------------------------------------------

ZENODO_DOI = "10.5281/zenodo.19813387"
ZENODO_RECORD_ID = "19813387"  # numeric record ID extracted from the DOI
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files"

TEST_FASTA = "yeast.contigs.fa.gz"
TEST_PAIRS = "yeast.pairs.gz"
TEST_ENZYME = ["DpnII", "HinfI"]

# Use a small but meaningful number of cycles for a functional smoke-test.
DEFAULT_TEST_CYCLES = 3
DEFAULT_TEST_LEVEL = 4


# ---------------------------------------------------------------------------
# Helpers
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
        # Extract version line (typically the last line)
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
    click.echo(f"  Detected CUDA runtime version {version[0]}.{version[1]}.{version[2]}.")  # e.g. (11, 8)

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

    # Check for nvcc availability
    _check_nvcc()


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        click.echo(f"\r  {pct:3d}%", nl=False)
    else:
        click.echo(f"\r  {block_num * block_size // 1024} KB", nl=False)


def _download(url: str, dest: pathlib.Path) -> None:
    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    click.echo()  # newline after progress bar


def _cmd_echo(cmd: list) -> None:
    """Print a command in shell-quoted form before running it."""
    click.echo("  $ " + " ".join(shlex.quote(str(a)) for a in cmd))


def _run_cmd(cmd: list, _record: list | None = None) -> None:
    """Print then execute a CLI command, streaming its output to the console."""
    _cmd_echo(cmd)
    if _record is not None:
        _record.append(" ".join(shlex.quote(str(a)) for a in cmd))
    subprocess.run([str(a) for a in cmd], check=True)


def _fetch_test_data(
    workdir: pathlib.Path,
    fasta_path_override: pathlib.Path | None = None,
    pairs_path_override: pathlib.Path | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Download or locate ``TEST_FASTA`` and ``TEST_PAIRS`` from Zenodo into *workdir*.

    If *fasta_path_override* or *pairs_path_override* are provided, use them instead of downloading.
    Otherwise, download from Zenodo (or reuse if already present in *workdir*).
    """
    # Use overrides if provided, otherwise default to workdir
    fasta_path = fasta_path_override if fasta_path_override else (workdir / TEST_FASTA)
    pairs_path = pairs_path_override if pairs_path_override else (workdir / TEST_PAIRS)

    # Validate that provided files exist
    if fasta_path_override and not fasta_path.exists():
        raise click.ClickException(f"FASTA file not found: {fasta_path}")
    if pairs_path_override and not pairs_path.exists():
        raise click.ClickException(f"Pairs file not found: {pairs_path}")

    # Download files only if not provided and not already present
    if not fasta_path_override and not fasta_path.exists():
        url = f"{ZENODO_BASE_URL}/{TEST_FASTA}"
        click.echo(f"  Fetching {TEST_FASTA} …")
        try:
            _download(url, fasta_path)
        except Exception as exc:
            raise click.ClickException(
                f"Failed to download {url}\n  {exc}\n\n"
                f"The test dataset is archived under DOI {ZENODO_DOI}.\n"
                "Check your internet connection or supply the files manually using --fasta and --pairs."
            ) from exc
    elif fasta_path_override or fasta_path.exists():
        click.echo(f"  Using FASTA from: {fasta_path}")

    if not pairs_path_override and not pairs_path.exists():
        url = f"{ZENODO_BASE_URL}/{TEST_PAIRS}"
        click.echo(f"  Fetching {TEST_PAIRS} …")
        try:
            _download(url, pairs_path)
        except Exception as exc:
            raise click.ClickException(
                f"Failed to download {url}\n  {exc}\n\n"
                f"The test dataset is archived under DOI {ZENODO_DOI}.\n"
                "Check your internet connection or supply the files manually using --fasta and --pairs."
            ) from exc
    elif pairs_path_override or pairs_path.exists():
        click.echo(f"  Using pairs from: {pairs_path}")

    return fasta_path, pairs_path


# ---------------------------------------------------------------------------
# Internal pipeline runner
# ---------------------------------------------------------------------------


def _run_test(
    workdir: pathlib.Path,
    device: int,
    level: int,
    cycles: int,
    fasta_path_override: pathlib.Path | None = None,
    pairs_path_override: pathlib.Path | None = None,
) -> list[str]:
    """Execute the full instagraal pipeline on the test dataset.

    Returns a list of shell-quoted commands that were executed, for the recap.
    """
    ran_cmds: list[str] = []
    # Resolve CLI entry points from the same environment as this process.
    # shutil.which() searches PATH and handles cases where the interpreter and
    # the entry-point scripts live in different directories (e.g. /usr/bin vs
    # /usr/local/bin in some Linux distributions).
    bin_dir = pathlib.Path(sys.executable).parent

    def cmd(name: str) -> pathlib.Path:
        found = shutil.which(name)
        if found:
            return pathlib.Path(found)
        return bin_dir / name

    # -- 1. GPU check --------------------------------------------------------
    click.echo("\n[1/7] Checking GPU …")
    _check_gpu(device)

    # -- 2. Download test data -----------------------------------------------
    if fasta_path_override or pairs_path_override:
        click.echo("\n[2/7] Locating test data …")
    else:
        click.echo(f"\n[2/7] Fetching test data from Zenodo (DOI: {ZENODO_DOI}) …")
    fasta_path, pairs_path = _fetch_test_data(workdir, fasta_path_override, pairs_path_override)

    # -- 3. instagraal-pre ---------------------------------------------------
    click.echo("\n[3/7] Running instagraal-pre …")
    pre_dir = workdir / "pre"
    pre_dir.mkdir(exist_ok=True)
    _run_cmd(
        [
            cmd("instagraal-pre"),
            fasta_path,
            pairs_path,
            "--enzyme",
            ",".join(TEST_ENZYME),
            "--output-dir",
            pre_dir,
        ],
        _record=ran_cmds,
    )

    # -- 4. instagraal (MCMC scaffolding) ------------------------------------
    click.echo("\n[4/7] Running instagraal …")
    mcmc_base_dir = workdir / "mcmc"
    mcmc_base_dir.mkdir(exist_ok=True)
    _run_cmd(
        [
            cmd("instagraal"),
            pre_dir,
            fasta_path,
            "--output-dir",
            mcmc_base_dir,
            "--level",
            level,
            "--cycles",
            cycles,
            "--save-matrix",
            "--bomb",
            "--device",
            device,
        ],
        _record=ran_cmds,
    )

    # instagraal writes: <output_folder>/<hic_folder_name>/test_mcmc_<level>/
    mcmc_out = mcmc_base_dir / pre_dir.name / f"test_mcmc_{level}"
    info_frags = mcmc_out / "info_frags.txt"
    if not info_frags.exists():
        raise click.ClickException(f"instagraal did not produce info_frags.txt at expected path:\n  {info_frags}")

    # -- 5. instagraal-polish ------------------------------------------------
    click.echo("\n[5/7] Running instagraal-polish …")
    polish_dir = workdir / "polished"
    polish_dir.mkdir(exist_ok=True)
    _run_cmd(
        [
            cmd("instagraal-polish"),
            "--input",
            info_frags,
            "--fasta",
            fasta_path,
            "--output-dir",
            polish_dir,
        ],
        _record=ran_cmds,
    )
    polished_fa = polish_dir / "polished_genome.fa"
    new_info_frags = polish_dir / "new_info_frags.txt"

    # -- 6. instagraal-post --------------------------------------------------
    click.echo("\n[6/7] Running instagraal-post …")
    post_dir = workdir / "post"
    post_dir.mkdir(exist_ok=True)
    _run_cmd(
        [
            cmd("instagraal-post"),
            pairs_path,
            new_info_frags,
            "--resolutions",
            "1000",
            "--output-dir",
            post_dir,
            "--no-balance",
        ],
        _record=ran_cmds,
    )

    # -- 7. instagraal-stats -------------------------------------------------
    click.echo("\n[7/7] Running instagraal-stats …")
    _run_cmd(
        [
            cmd("instagraal-stats"),
            fasta_path,
            polished_fa,
            "--labels",
            "Input assembly,Polished assembly",
        ],
        _record=ran_cmds,
    )

    return ran_cmds


# ---------------------------------------------------------------------------
# Click command
# ---------------------------------------------------------------------------


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(VERSION_NUMBER, "-V", "--version")
@click.option(
    "-w",
    "--workdir",
    default=None,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help=(
        "Working directory for downloaded files and pipeline outputs. "
        "Defaults to a temporary directory that is deleted on exit "
        "(unless --keep is set)."
    ),
)
@click.option(
    "--keep",
    is_flag=True,
    default=False,
    help="Keep the working directory after the test completes (useful for inspection).",
)
@click.option(
    "--fasta",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help=("Path to manually downloaded FASTA file. If provided, skips downloading from Zenodo. Useful for offline environments."),
)
@click.option(
    "--pairs",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path),
    help=("Path to manually downloaded pairs file. If provided, skips downloading from Zenodo. Useful for offline environments."),
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
    default=DEFAULT_TEST_LEVEL,
    show_default=True,
    type=int,
    help="Contact-map resolution level passed to instagraal.",
)
@click.option(
    "-n",
    "--cycles",
    default=DEFAULT_TEST_CYCLES,
    show_default=True,
    type=int,
    help="Number of MCMC cycles passed to instagraal.",
)
def main(
    workdir: pathlib.Path | None,
    keep: bool,
    fasta: pathlib.Path | None,
    pairs: pathlib.Path | None,
    device: int,
    level: int,
    cycles: int,
) -> None:
    """End-to-end functional test for instaGRAAL.

    \b
    Steps
    -----
      1. Check that a CUDA-capable GPU is accessible.
      2. Download the test dataset (yeast in silico assembly + Hi-C pairs)
         from Zenodo (DOI: 10.5281/zenodo.19813387), or use locally provided
         files if --fasta and --pairs are specified.
      3. instagraal-pre   - digest FASTA and bin Hi-C pairs into fragments.
      4. instagraal       - MCMC-based scaffolding.
      5. instagraal-polish - full polishing pipeline.
      6. instagraal-post   - remap pairs to new assembly, write mcool.
      7. instagraal-stats  - compare input vs. polished assembly statistics.

    The test uses S. cerevisiae as the model organism (enzyme: DpnII).
    """
    _tmp_dir: str | None = None

    if workdir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="instagraal_test_")
        workdir = pathlib.Path(_tmp_dir)
    else:
        workdir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Working directory: {workdir}")

    _start = time.monotonic()
    try:
        ran_cmds = _run_test(
            workdir=workdir,
            device=device,
            level=level,
            cycles=cycles,
            fasta_path_override=fasta,
            pairs_path_override=pairs,
        )
        _elapsed = time.monotonic() - _start
        click.echo(f"\n[instagraal-test] ALL STEPS PASSED  (total time: {_elapsed:.1f}s).")
        click.echo("\nCommands executed (for reproducibility):")
        click.echo("\n```")
        for c in ran_cmds:
            click.echo(c)
        click.echo("```")
    finally:
        if _tmp_dir is not None and not keep:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        else:
            click.echo(f"\nOutput files retained in: {workdir}")
