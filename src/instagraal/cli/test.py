"""Click CLI for the ``instagraal-test`` end-to-end functional test command."""

import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request

import click

from ..version import __version__ as VERSION_NUMBER

# ---------------------------------------------------------------------------
# Zenodo configuration
# Update ZENODO_RECORD_ID once the dataset is published.
# ---------------------------------------------------------------------------

ZENODO_DOI = "10.5281/zenodo.19711358"
ZENODO_RECORD_ID = "19711358"  # numeric record ID extracted from the DOI
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


def _check_gpu(device: int) -> None:
    """Verify that pycuda can initialise the requested CUDA device."""
    try:
        import pycuda.driver as cuda  # noqa: PLC0415
    except ImportError as exc:
        raise click.ClickException("pycuda is not installed.  instagraal requires a CUDA-capable GPU.") from exc

    try:
        cuda.init()
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"CUDA initialisation failed: {exc}") from exc

    n_devices = cuda.Device.count()
    if n_devices == 0:
        raise click.ClickException("No CUDA devices found.  instagraal requires at least one GPU.")
    if device >= n_devices:
        raise click.ClickException(
            f"Requested device {device} but only {n_devices} device(s) available " f"(0–{n_devices - 1})."
        )

    dev = cuda.Device(device)
    click.echo(f"  GPU {device}: {dev.name()} " f"({dev.total_memory() // 1024 ** 2} MB)")


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


def _run_cmd(cmd: list) -> None:
    """Print then execute a CLI command, streaming its output to the console."""
    _cmd_echo(cmd)
    subprocess.run([str(a) for a in cmd], check=True)


def _fetch_test_data(workdir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Download ``TEST_FASTA`` and ``TEST_PAIRS`` from Zenodo into *workdir*."""
    fasta_path = workdir / TEST_FASTA
    pairs_path = workdir / TEST_PAIRS

    for filename, dest in [(TEST_FASTA, fasta_path), (TEST_PAIRS, pairs_path)]:
        if dest.exists():
            click.echo(f"  {filename} already present, skipping download.")
            continue
        url = f"{ZENODO_BASE_URL}/{filename}"
        click.echo(f"  Fetching {filename} …")
        try:
            _download(url, dest)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(
                f"Failed to download {url}\n  {exc}\n\n"
                f"The test dataset is archived under DOI {ZENODO_DOI}.\n"
                "Check your internet connection or supply the files manually."
            ) from exc

    return fasta_path, pairs_path


# ---------------------------------------------------------------------------
# Internal pipeline runner
# ---------------------------------------------------------------------------


def _run_test(
    workdir: pathlib.Path,
    device: int,
    level: int,
    cycles: int,
) -> None:
    """Execute the full instagraal pipeline on the test dataset."""
    # Resolve CLI entry points from the same environment as this process.
    bin_dir = pathlib.Path(sys.executable).parent

    def cmd(name: str) -> pathlib.Path:
        return bin_dir / name

    # -- 1. GPU check --------------------------------------------------------
    click.echo("\n[1/6] Checking GPU …")
    _check_gpu(device)

    # -- 2. Download test data -----------------------------------------------
    click.echo(f"\n[2/6] Fetching test data from Zenodo (DOI: {ZENODO_DOI}) …")
    fasta_path, pairs_path = _fetch_test_data(workdir)

    # -- 3. instagraal-pre ---------------------------------------------------
    click.echo("\n[3/6] Running instagraal-pre …")
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
        ]
    )

    # -- 4. instagraal (MCMC scaffolding) ------------------------------------
    click.echo("\n[4/6] Running instagraal …")
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
        ]
    )

    # instagraal writes: <output_folder>/<hic_folder_name>/test_mcmc_<level>/
    mcmc_out = mcmc_base_dir / pre_dir.name / f"test_mcmc_{level}"
    info_frags = mcmc_out / "info_frags.txt"
    if not info_frags.exists():
        raise click.ClickException(f"instagraal did not produce info_frags.txt at expected path:\n  {info_frags}")

    # -- 5. instagraal-polish ------------------------------------------------
    click.echo("\n[5/6] Running instagraal-polish …")
    polish_dir = workdir / "polished"
    polish_dir.mkdir(exist_ok=True)
    _run_cmd(
        [
            cmd("instagraal-polish"),
            "--mode",
            "polishing",
            "--input",
            info_frags,
            "--fasta",
            fasta_path,
            "--output-dir",
            polish_dir,
        ]
    )
    polished_fa = polish_dir / "polished_genome.fa"

    # -- 6. instagraal-stats -------------------------------------------------
    click.echo("\n[6/6] Running instagraal-stats …")
    _run_cmd(
        [
            cmd("instagraal-stats"),
            fasta_path,
            polished_fa,
            "--labels",
            "Input assembly,Polished assembly",
        ]
    )


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
         from Zenodo (DOI: 10.5281/zenodo.19711358).
      3. instagraal-pre   – digest FASTA and bin Hi-C pairs into fragments.
      4. instagraal       – MCMC-based scaffolding.
      5. instagraal-polish – full polishing pipeline.
      6. instagraal-stats  – compare input vs. polished assembly statistics.

    The test uses S. cerevisiae as the model organism (enzyme: DpnII).
    """
    _tmp_dir: str | None = None

    if workdir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="instagraal_test_")
        workdir = pathlib.Path(_tmp_dir)

    click.echo(f"Working directory: {workdir}")

    try:
        _run_test(workdir=workdir, device=device, level=level, cycles=cycles)
        click.echo("\n[instagraal-test] ALL STEPS PASSED.")
    finally:
        if _tmp_dir is not None and not keep:
            shutil.rmtree(_tmp_dir, ignore_errors=True)
        else:
            click.echo(f"\nOutput files retained in: {workdir}")
