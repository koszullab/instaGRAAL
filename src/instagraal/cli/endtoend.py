"""Click CLI for the ``instagraal-endtoend`` full pipeline command."""

import contextlib
import dataclasses
import gzip
import pathlib
import shlex
import shutil
import subprocess
import sys
import time

import click
from click.core import ParameterSource

from ..pre import default_cool_name
from ..version import __version__ as VERSION_NUMBER

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_LEVEL = 4
DEFAULT_CYCLES = 100  # MCMC cycles when no refine follows (instaGRAAL's own default)
DEFAULT_CYCLES_WITH_REFINE = 10  # refine only uses the MCMC layout to decide its second pass
DEFAULT_RESOLUTIONS = "1000"
DEFAULT_MODE = "default"
# instagraal-refine scores every pair of contig ends in dense matrices: about 256 bytes
# times the square of the contig count (some 6 GB at 5,000 contigs)
REFINE_MAX_CONTIGS = 5000
FINAL_GENOME_NAME = "final_genome.fa"
FINAL_INFO_FRAGS_NAME = "final_info_frags.txt"


# ---------------------------------------------------------------------------
# Modes: the steps run between instagraal-pre and instagraal-post
# ---------------------------------------------------------------------------

STEPS = ("mcmc", "bomb", "polish", "refine", "extend", "noextend")
PRESETS = {
    "default": "mcmc,refine",
    "cpu": "refine,extend",
    "legacy": "bomb,mcmc,polish",
}


@dataclasses.dataclass(frozen=True)
class Plan:
    """What a run does: MCMC scaffolding (from bins with ``bomb``), polish, refine and refine's ``--extend``."""

    mcmc: bool = False
    bomb: bool = False
    polish: bool = False
    refine: bool = False
    extend: str = "auto"

    def __str__(self) -> str:
        steps = ",".join(s for s in ("bomb", "mcmc", "polish", "refine") if getattr(self, s))
        return f"{steps} (extend: {self.extend})" if self.refine else steps


def parse_mode(text: str) -> Plan:
    """The plan a ``--mode`` value asks for: a preset alone, or steps separated by commas.

    Names are case-insensitive and steps run in pipeline order whatever their order in
    *text*. Raises ``ValueError`` saying what is wrong.
    """
    tokens = [t.strip().lower() for t in text.split(",") if t.strip()]
    if not tokens:
        raise ValueError(f"no step given; use a preset ({', '.join(PRESETS)}) or steps among {', '.join(STEPS)}")
    preset = next((t for t in tokens if t in PRESETS), None)
    if preset is not None:
        if len(tokens) > 1:
            raise ValueError(f"the preset '{preset}' stands alone; to change it, list its steps ({PRESETS[preset]})")
        tokens = PRESETS[preset].split(",")
    unknown = sorted({t for t in tokens if t not in STEPS})
    if unknown:
        raise ValueError(f"unknown step(s): {', '.join(unknown)}; steps are {', '.join(STEPS)}, presets {', '.join(PRESETS)}")
    steps = set(tokens)
    if "bomb" in steps and "mcmc" not in steps:
        raise ValueError("bomb is an MCMC option: add mcmc")
    if "polish" in steps and "mcmc" not in steps:
        raise ValueError("polish works on the MCMC scaffolds: add mcmc")
    if steps & {"extend", "noextend"} and "refine" not in steps:
        raise ValueError("extend and noextend are refine options: add refine")
    if {"extend", "noextend"} <= steps:
        raise ValueError("extend and noextend exclude each other")
    if not steps & {"mcmc", "refine"}:
        raise ValueError("nothing scaffolds: add mcmc, refine or both")
    extend = "on" if "extend" in steps else "off" if "noextend" in steps else "auto"
    return Plan(mcmc="mcmc" in steps, bomb="bomb" in steps, polish="polish" in steps, refine="refine" in steps, extend=extend)


def plan_notes(plan: Plan) -> list[str]:
    """What a plan runs that does not change its assembly, or does less than its name suggests."""
    notes = []
    if plan.mcmc and plan.refine and plan.extend != "auto":
        notes.append(f"with extend {plan.extend}, refine does not use the MCMC layout: the MCMC run does not change the assembly")
    if plan.polish and plan.refine:
        notes.append("refine gives the same assembly with or without polish")
    if plan.refine and not plan.mcmc and plan.extend == "auto":
        notes.append(
            "without mcmc, refine has no layout to check its second pass against, so that pass does not run "
            "(add extend to force it)"
        )
    return notes


def _mode_callback(ctx: click.Context, param: click.Parameter, value: str) -> Plan:
    try:
        return parse_mode(value)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from None


def _count_contigs(fasta: pathlib.Path) -> int:
    opener = gzip.open if str(fasta).endswith((".gz", ".bgz")) else open
    with opener(fasta, "rt") as fh:
        return sum(1 for line in fh if line.startswith(">"))


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
        raise click.ClickException(
            "pycuda is not installed: modes with mcmc need it and a CUDA GPU (pip install 'instagraal[gpu]', which "
            "needs the CUDA toolkit). Use --mode cpu to run without them."
        ) from exc
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


def _install_final(final_fa: pathlib.Path, final_info_frags: pathlib.Path, output_dir: pathlib.Path, dry_run: bool) -> None:
    """Copy the final assembly and its layout to fixed names in *output_dir*, whatever the mode."""
    for source, target in ((final_fa, output_dir / FINAL_GENOME_NAME), (final_info_frags, output_dir / FINAL_INFO_FRAGS_NAME)):
        click.echo(f"  {target}  <-  {source}")
        if not dry_run:
            shutil.copyfile(source, target)


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
    plan: Plan,
    dry_run: bool = False,
) -> list[str]:
    """Execute the instaGRAAL pipeline that *plan* describes on user-supplied data.

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

    steps = (["gpu"] if plan.mcmc else []) + ["pre"] + (["mcmc", "polish"] if plan.mcmc else [])
    steps += (["refine"] if plan.refine else []) + ["post", "stats"]

    def label(step: str, text: str) -> str:
        return f"[{steps.index(step) + 1}/{len(steps)}] {text}"

    # -- GPU check (only the MCMC needs one) ----------------------------------
    if plan.mcmc:
        with _step(label("gpu", "Checking GPU …")):
            if not dry_run:
                _check_gpu(device)
            else:
                click.echo("  (skipped — dry run)")

    # -- instagraal-pre --------------------------------------------------------
    with _step(label("pre", "Running instagraal-pre …")):
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

    final_fa = final_info_frags = pathlib.Path()
    final_label = ""
    if plan.mcmc:
        # -- instagraal (MCMC scaffolding) -------------------------------------
        with _step(label("mcmc", "Running instagraal …")):
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
            if plan.bomb:
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

        # -- instagraal-polish -------------------------------------------------
        # Without polish, ``instagraal-polish --mode fasta`` writes the scaffolds
        # as instagraal left them (still size-filtered), plus the matching
        # new_info_frags.txt so that instagraal-post can lift the pairs over.
        text = "Running instagraal-polish …" if plan.polish else "Skipping polishing: writing scaffolds as is …"
        with _step(label("polish", text)):
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
            if not plan.polish:
                polish_cmd += ["--mode", "fasta"]
            # Passed even when empty: instagraal-post is told len(junction), so
            # instagraal-polish must not fall back to its own default junction.
            polish_cmd += ["--junction", junction]
            if criterion is not None and plan.polish:
                polish_cmd += ["--criterion", criterion]
            _run_cmd(polish_cmd, _record=ran_cmds, dry_run=dry_run)
        final_fa = polish_dir / "polished_genome.fa"
        final_info_frags = polish_dir / "new_info_frags.txt"
        final_label = "Polished assembly" if plan.polish else "Scaffolded assembly (not polished)"

    if plan.refine:
        # -- instagraal-refine -------------------------------------------------
        with _step(label("refine", "Running instagraal-refine …")):
            refine_dir = output_dir / "refined"
            refine_dir.mkdir(exist_ok=True)
            pre_cool = pre_dir / f"{cool_name or default_cool_name(pairs)}.cool"
            refine_cmd: list = [cmd("instagraal-refine")]
            if plan.mcmc:  # the MCMC layout, which --extend auto checks before its second pass
                refine_cmd += ["--info-frags", final_info_frags]
            refine_cmd += [
                "--cool",
                pre_cool,
                "--fasta",
                fasta,
                "--output-dir",
                refine_dir,
                # refine inserts the same junction between contigs as polish does
                # between bins, so instagraal-post's len(junction) still holds
                "--junction",
                junction,
                "--extend",
                plan.extend,
            ]
            _run_cmd(refine_cmd, _record=ran_cmds, dry_run=dry_run)
        final_fa = refine_dir / "refined_genome.fa"
        final_info_frags = refine_dir / "refined_info_frags.txt"
        final_label = "Refined assembly"

    click.echo("\nFinal assembly:")
    _install_final(final_fa, final_info_frags, output_dir, dry_run)

    # -- instagraal-post --------------------------------------------------------
    with _step(label("post", "Running instagraal-post …")):
        post_dir = output_dir / "post"
        post_dir.mkdir(exist_ok=True)
        post_cmd = [
            cmd("instagraal-post"),
            pairs,
            final_info_frags,
            "--resolutions",
            resolutions,
            "--output-dir",
            post_dir,
            "--junction",
            len(junction),
        ]
        if not balance:
            post_cmd.append("--no-balance")
        if balance_args is not None:
            post_cmd += ["--balance-args", balance_args]
        if cool_name is not None:
            post_cmd += ["--cool-name", cool_name]
        _run_cmd(post_cmd, _record=ran_cmds, dry_run=dry_run)

    # -- instagraal-stats -------------------------------------------------------
    with _step(label("stats", "Running instagraal-stats …")):
        _run_cmd(
            [
                cmd("instagraal-stats"),
                fasta,
                final_fa,
                "--labels",
                f"Input assembly,{final_label}",
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
    "-m",
    "--mode",
    default=DEFAULT_MODE,
    show_default=True,
    callback=_mode_callback,
    help=(
        "Steps to run between instagraal-pre and instagraal-post: a preset, or steps separated by commas, in any "
        "order. Steps: mcmc (MCMC scaffolding, needs a GPU), bomb (start the MCMC from bins rather than the "
        "contigs), polish, refine (re-join whole contigs from the contacts next to their ends), extend/noextend "
        "(force refine's second, long-range pass on or off; by default it runs only if refine confirms the MCMC "
        "layout). Presets: default = mcmc,refine; cpu = refine,extend (no GPU, the most contiguous assemblies); "
        "legacy = bomb,mcmc,polish (the pipeline before refine)."
    ),
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
    default=None,
    type=int,
    help=(
        "Number of MCMC iterations per bin. A high number has diminishing returns but a minimum is needed for "
        f"convergence.  [default: {DEFAULT_CYCLES_WITH_REFINE} when the mode includes refine, which only uses the "
        f"MCMC to decide its second pass; {DEFAULT_CYCLES} otherwise]"
    ),
)
@click.option(
    "-r",
    "--resolutions",
    default=DEFAULT_RESOLUTIONS,
    show_default=True,
    help=(
        "Resolution(s) of the output .mcool, in bp, comma-separated (e.g. '1000,5000,10000'), passed to "
        "instagraal-post: each must be a multiple of a smaller one, and a single value gives one resolution."
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
@click.option("-b", "--bomb/--no-bomb", default=None, hidden=True, help="Deprecated: use --mode.")
@click.option("--polish/--no-polish", default=None, hidden=True, help="Deprecated: use --mode.")
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
        "Junction sequence inserted between stitched bins by instagraal-polish, and between contigs by "
        "instagraal-refine. Its length is automatically forwarded to instagraal-post (--junction). "
    ),
)
@click.option(
    "--criterion",
    default=None,
    help="Reorientation criterion passed to instagraal-polish: blocks (default) or sequences.",
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
    mode: Plan,
    device: int,
    level: int,
    cycles: int | None,
    resolutions: str,
    coverage_std: float,
    neighborhood: int,
    bomb: bool | None,
    polish: bool | None,
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
      instagraal-pre    — digest FASTA and bin Hi-C pairs into fragments.
      Then, as --mode says:
        mcmc            — check the GPU, then scaffold with the MCMC
                          (instagraal), from bins with bomb, else from the
                          contigs;
        polish          — instagraal-polish; without it, the MCMC scaffolds
                          are written as they are;
        refine          — instagraal-refine: re-join whole contigs from the
                          contacts next to their ends (extend/noextend force
                          its second, long-range pass on or off).
      instagraal-post   — remap pairs to the final assembly, write .mcool.
      instagraal-stats  — compare the input and final assemblies.

    \b
    Modes
    -----
      default   mcmc,refine        needs a GPU
      cpu       refine,extend      no GPU; the most contiguous assemblies
      legacy    bomb,mcmc,polish   the pipeline before refine

    \b
    Required inputs
    ---------------
      FASTA   Genome assembly FASTA file (plain or .gz).
      PAIRS   Hi-C pairs file in 4DN pairs format (plain or .gz).

    All intermediate and final results are written under OUTPUT_DIR; the final
    assembly is also copied to OUTPUT_DIR/final_genome.fa (and its layout to
    final_info_frags.txt), whatever the mode.
    """
    ctx = click.get_current_context()
    mode_given = ctx.get_parameter_source("mode") is not ParameterSource.DEFAULT
    plan = mode
    if bomb is not None or polish is not None:
        # the flags of earlier versions keep their meaning: they adjust the pipeline they adjusted
        if mode_given:
            raise click.UsageError("--bomb/--no-bomb and --polish/--no-polish are replaced by --mode: give --mode alone.")
        legacy = parse_mode("legacy")
        plan = dataclasses.replace(
            legacy, bomb=legacy.bomb if bomb is None else bomb, polish=legacy.polish if polish is None else polish
        )
        mode_given = True
        click.echo(f"Warning: --bomb/--no-bomb and --polish/--no-polish are deprecated; this is --mode {plan}.", err=True)

    if plan.refine:
        n_contigs = _count_contigs(fasta)
        if n_contigs > REFINE_MAX_CONTIGS:
            what = (
                f"{n_contigs:,} contigs, more than instagraal-refine handles ({REFINE_MAX_CONTIGS:,}: it compares every "
                "contig end with every other, and its memory grows with the square of the number of contigs)"
            )
            if mode_given:
                raise click.UsageError(f"{what}. Use --mode legacy, or another mode without refine.")
            click.echo(f"Warning: {what}; running --mode legacy ({parse_mode('legacy')}) instead.", err=True)
            plan = parse_mode("legacy")

    if cycles is None:
        cycles = DEFAULT_CYCLES_WITH_REFINE if plan.refine else DEFAULT_CYCLES

    enzymes = [e.strip() for e in enzyme.split(",") if e.strip()]
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Output directory : {output_dir.resolve()}")
    click.echo(f"FASTA            : {fasta.resolve()}")
    click.echo(f"Pairs            : {pairs.resolve()}")
    click.echo(f"Enzyme(s)        : {', '.join(enzymes)}")
    click.echo(f"Mode             : {plan}" + ("" if plan.mcmc else "  (no GPU needed)"))
    for note in plan_notes(plan):
        click.echo(f"Note: {note}.")
    if not plan.mcmc and ctx.get_parameter_source("cycles") is not ParameterSource.DEFAULT:
        click.echo("Note: --cycles is ignored: this mode runs no MCMC.")
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
        plan=plan,
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
