"""Tests for the ``instagraal-endtoend`` CLI command.

All tests mock ``_run_endtoend`` or use its dry run, so no GPU or heavy computation is required.
"""

import pathlib
import shlex
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import instagraal.cli.endtoend as endtoend_module
from instagraal.cli.endtoend import PRESETS, Plan, _install_final, main, parse_mode, plan_notes

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"

REF_FASTA = TEST_DATA / "yeast.contigs.fa.gz"
REF_PAIRS = TEST_DATA / "yeast.pairs.gz"

needs_data = pytest.mark.skipif(not REF_FASTA.exists() or not REF_PAIRS.exists(), reason="Test data files not present")

DEFAULT_PLAN = Plan(mcmc=True, refine=True)
CPU_PLAN = Plan(refine=True, extend="on")
LEGACY_PLAN = Plan(mcmc=True, bomb=True, polish=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _invoke(args: list) -> object:
    """Invoke the CLI with the given *args* and return the Click result."""
    runner = CliRunner()
    return runner.invoke(main, [str(a) for a in args])


def _base_args(**overrides) -> list:
    """Return a minimal valid argument list, with optional overrides."""
    defaults = {
        "fasta": REF_FASTA,
        "pairs": REF_PAIRS,
        "enzyme": "DpnII",
    }
    defaults.update(overrides)
    return [defaults["fasta"], defaults["pairs"], "--enzyme", defaults["enzyme"]]


def _captured_kwargs(args: list, tmp_path) -> tuple[object, dict]:
    """Run the CLI with ``_run_endtoend`` mocked; return the result and the arguments it got."""
    captured: dict = {}

    def _capture(*a, **kwargs):
        captured.update(kwargs)
        return []

    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_capture):
        result = _invoke([*_base_args(), "--output-dir", tmp_path, *args])
    return result, captured


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help():
    result = _invoke(["--help"])
    assert result.exit_code == 0
    assert "FASTA" in result.output
    assert "PAIRS" in result.output
    assert "--enzyme" in result.output
    assert "--mode" in result.output
    # the flags --mode replaces are hidden
    assert "--no-bomb" not in result.output and "--no-polish" not in result.output


# ---------------------------------------------------------------------------
# Missing required arguments
# ---------------------------------------------------------------------------


def test_missing_fasta():
    runner = CliRunner()
    result = runner.invoke(main, ["--enzyme", "DpnII"])
    assert result.exit_code != 0


def test_missing_enzyme():
    result = _invoke([REF_FASTA, REF_PAIRS])
    assert result.exit_code != 0
    assert "enzyme" in result.output.lower() or "missing" in result.output.lower()


def test_nonexistent_fasta(tmp_path):
    result = _invoke([tmp_path / "no.fa", REF_PAIRS, "--enzyme", "DpnII"])
    assert result.exit_code != 0


def test_nonexistent_pairs(tmp_path):
    result = _invoke([REF_FASTA, tmp_path / "no.pairs", "--enzyme", "DpnII"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# --mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,plan",
    [
        ("default", DEFAULT_PLAN),
        ("CPU", CPU_PLAN),
        ("legacy", LEGACY_PLAN),
        ("refine,mcmc", DEFAULT_PLAN),  # the order does not matter
        (" mcmc , Polish ", Plan(mcmc=True, polish=True)),
        ("mcmc", Plan(mcmc=True)),
        ("refine", Plan(refine=True)),
        ("mcmc,refine,noextend", Plan(mcmc=True, refine=True, extend="off")),
        ("bomb,mcmc,polish,refine,extend", Plan(mcmc=True, bomb=True, polish=True, refine=True, extend="on")),
    ],
)
def test_parse_mode(text, plan):
    assert parse_mode(text) == plan


def test_presets_are_their_steps():
    for name, steps in PRESETS.items():
        assert parse_mode(name) == parse_mode(steps)


@pytest.mark.parametrize(
    "text,message",
    [
        ("", "no step given"),
        (" , ", "no step given"),
        ("cpu,noextend", "stands alone"),
        ("mcmc,gpu", "unknown step"),
        ("bomb", "bomb is an MCMC option"),
        ("bomb,refine", "bomb is an MCMC option"),
        ("polish,refine", "polish works on the MCMC scaffolds"),
        ("mcmc,extend", "refine options"),
        ("refine,extend,noextend", "exclude each other"),
    ],
)
def test_parse_mode_rejects(text, message):
    with pytest.raises(ValueError, match=message):
        parse_mode(text)


def test_plan_reads_as_its_steps():
    assert str(DEFAULT_PLAN) == "mcmc,refine (extend: auto)"
    assert str(CPU_PLAN) == "refine (extend: on)"
    assert str(LEGACY_PLAN) == "bomb,mcmc,polish"


@pytest.mark.parametrize(
    "plan,expected",
    [
        (DEFAULT_PLAN, []),
        (LEGACY_PLAN, []),
        (CPU_PLAN, []),
        (Plan(mcmc=True, refine=True, extend="on"), ["does not change the assembly"]),
        (Plan(mcmc=True, polish=True, refine=True), ["with or without polish"]),
        (Plan(refine=True), ["that pass does not run"]),
    ],
)
def test_plan_notes(plan, expected):
    notes = plan_notes(plan)
    assert len(notes) == len(expected)
    for note, words in zip(notes, expected, strict=True):
        assert words in note


@needs_data
@pytest.mark.parametrize(
    "value,plan", [("cpu", CPU_PLAN), ("legacy", LEGACY_PLAN), ("mcmc,polish,refine", Plan(mcmc=True, polish=True, refine=True))]
)
def test_mode_is_forwarded(value, plan, tmp_path):
    result, captured = _captured_kwargs(["--mode", value], tmp_path)
    assert result.exit_code == 0, result.output
    assert captured["plan"] == plan
    assert f"Mode             : {plan}" in result.output


@needs_data
def test_bad_mode_is_a_usage_error(tmp_path):
    result, captured = _captured_kwargs(["--mode", "polish"], tmp_path)
    assert result.exit_code == 2
    assert "polish works on the MCMC scaffolds" in result.output
    assert not captured


@needs_data
def test_notes_are_printed(tmp_path):
    result, _captured = _captured_kwargs(["--mode", "mcmc,refine,extend"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "Note: with extend on, refine does not use the MCMC layout" in result.output


@needs_data
@pytest.mark.parametrize(
    "args,cycles",
    [
        ([], 10),
        (["--mode", "legacy"], 100),
        (["--mode", "mcmc"], 100),
        (["--mode", "bomb,mcmc,refine"], 10),
        (["--cycles", "5"], 5),
        (["--mode", "legacy", "--cycles", "5"], 5),
    ],
)
def test_cycles_default_follows_the_mode(args, cycles, tmp_path):
    result, captured = _captured_kwargs(args, tmp_path)
    assert result.exit_code == 0, result.output
    assert captured["cycles"] == cycles


@needs_data
def test_cycles_without_mcmc_are_ignored(tmp_path):
    result, _captured = _captured_kwargs(["--mode", "cpu", "--cycles", "5"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "--cycles is ignored" in result.output
    assert "(no GPU needed)" in result.output


@needs_data
@pytest.mark.parametrize(
    "flags,plan",
    [
        (["--no-polish"], Plan(mcmc=True, bomb=True)),
        (["--no-bomb"], Plan(mcmc=True, polish=True)),
        (["--no-bomb", "--no-polish"], Plan(mcmc=True)),
        (["-b", "--polish"], LEGACY_PLAN),
    ],
)
def test_deprecated_flags_adjust_the_legacy_pipeline(flags, plan, tmp_path):
    result, captured = _captured_kwargs(flags, tmp_path)
    assert result.exit_code == 0, result.output
    assert captured["plan"] == plan
    assert "deprecated" in result.output


@needs_data
def test_deprecated_flags_with_a_mode_are_refused(tmp_path):
    result, captured = _captured_kwargs(["--mode", "cpu", "--no-bomb"], tmp_path)
    assert result.exit_code == 2
    assert "replaced by --mode" in result.output
    assert not captured


@needs_data
def test_too_many_contigs_fall_back_to_legacy_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(endtoend_module, "REFINE_MAX_CONTIGS", 10)  # the yeast assembly has 146
    result, captured = _captured_kwargs([], tmp_path)
    assert result.exit_code == 0, result.output
    assert captured["plan"] == LEGACY_PLAN
    assert captured["cycles"] == 100
    assert "146 contigs" in result.output and "running --mode legacy" in result.output


@needs_data
@pytest.mark.parametrize("value", ["cpu", "default", "mcmc,refine"])
def test_too_many_contigs_for_an_explicit_refine_are_refused(value, tmp_path, monkeypatch):
    monkeypatch.setattr(endtoend_module, "REFINE_MAX_CONTIGS", 10)
    result, captured = _captured_kwargs(["--mode", value], tmp_path)
    assert result.exit_code == 2
    assert "146 contigs" in result.output and "--mode legacy" in result.output
    assert not captured


@needs_data
def test_cpu_mode_needs_no_gpu_check(tmp_path):
    result = _invoke([*_base_args(), "--output-dir", tmp_path, "--mode", "cpu", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Checking GPU" not in result.output
    assert "[1/4] Running instagraal-pre" in result.output
    assert "[2/4] Running instagraal-refine" in result.output


# ---------------------------------------------------------------------------
# Successful invocation (mocked pipeline)
# ---------------------------------------------------------------------------


def _mock_run_endtoend(*args, **kwargs) -> list[str]:
    return [
        "instagraal-pre ...",
        "instagraal ...",
        "instagraal-refine ...",
        "instagraal-post ...",
        "instagraal-stats ...",
    ]


@needs_data
def test_success_default_options(tmp_path):
    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_mock_run_endtoend):
        result = _invoke([*_base_args(), "--output-dir", tmp_path])
    assert result.exit_code == 0, result.output
    assert "ALL STEPS COMPLETED" in result.output


@needs_data
def test_recap_commands_printed(tmp_path):
    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_mock_run_endtoend):
        result = _invoke([*_base_args(), "--output-dir", tmp_path])
    assert "instagraal-pre" in result.output
    assert "instagraal-stats" in result.output


# ---------------------------------------------------------------------------
# Option forwarding
# ---------------------------------------------------------------------------


@needs_data
@pytest.mark.parametrize(
    "extra_args,expected_kwargs",
    [
        (["--no-save-matrix"], {"save_matrix": False}),
        (["--no-balance"], {"balance": False}),
        (["--cycles", "5"], {"cycles": 5}),
        (["--level", "3"], {"level": 3}),
        (["--device", "1"], {"device": 1}),
        (["--min-scaffold-size", "10"], {"min_scaffold_size": 10}),
        (["--min-scaffold-length", "5000"], {"min_scaffold_length": 5000}),
        (["--junction", "AATTCC"], {"junction": "AATTCC"}),
        (["--enzyme", "HinfI"], {"enzymes": ["HinfI"]}),
        (["--enzyme", "DpnII,HinfI"], {"enzymes": ["DpnII", "HinfI"]}),
        (["--quiet"], {"quiet": True}),
        (["--debug"], {"debug": True}),
        (["--simple"], {"simple": True}),
        (["--save-pickle"], {"save_pickle": True}),
        (["--circular"], {"circular": True}),
        (["-m", "legacy"], {"plan": LEGACY_PLAN}),
    ],
)
def test_option_forwarding(extra_args, expected_kwargs, tmp_path):
    """Each CLI option is forwarded to ``_run_endtoend`` with the right value."""
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return []

    base = [REF_FASTA, REF_PAIRS, "--enzyme", "DpnII", "--output-dir", str(tmp_path)]
    # Replace enzyme option if the parametrize set provides one
    if "--enzyme" in extra_args:
        idx = base.index("--enzyme")
        base = base[:idx] + base[idx + 2 :]  # remove existing --enzyme + value

    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_capture):
        result = _invoke(base + [str(a) for a in extra_args])

    assert result.exit_code == 0, result.output
    for key, val in expected_kwargs.items():
        assert captured.get(key) == val, f"{key}: expected {val!r}, got {captured.get(key)!r}"


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


@needs_data
def test_defaults(tmp_path):
    result, captured = _captured_kwargs([], tmp_path)
    assert result.exit_code == 0, result.output
    assert captured["plan"] == DEFAULT_PLAN
    assert captured["level"] == 4
    assert captured["cycles"] == 10
    assert captured["save_matrix"] is True
    assert captured["balance"] is True
    assert captured["device"] == 0
    assert captured["min_scaffold_size"] == 0
    assert captured["min_scaffold_length"] == 0
    assert captured["junction"] == "NNNNNN"


# ---------------------------------------------------------------------------
# Commands built by _run_endtoend (dry run: nothing is executed)
# ---------------------------------------------------------------------------


def _dry_run_commands(tmp_path, **overrides) -> list[str]:
    from instagraal.cli.endtoend import _run_endtoend

    kwargs = {
        "fasta": REF_FASTA,
        "pairs": REF_PAIRS,
        "enzymes": ["DpnII"],
        "output_dir": tmp_path,
        "device": 0,
        "level": 4,
        "cycles": 10,
        "resolutions": "1000",
        "circular": False,
        "save_matrix": True,
        "save_pickle": False,
        "simple": False,
        "quiet": False,
        "debug": False,
        "balance": True,
        "balance_args": None,
        "coverage_std": 1.0,
        "neighborhood": 5,
        "min_scaffold_size": 0,
        "min_scaffold_length": 0,
        "junction": "NNNNNN",
        "criterion": None,
        "cool_name": None,
        "plan": LEGACY_PLAN,
        "dry_run": True,
    }
    kwargs.update(overrides)
    return _run_endtoend(**kwargs)


def _names(commands: list[str]) -> list[str]:
    return [pathlib.Path(shlex.split(c)[0]).name for c in commands]


def _command(commands: list[str], name: str) -> list[str]:
    matches = [shlex.split(c) for c in commands if pathlib.Path(shlex.split(c)[0]).name == name]
    assert len(matches) == 1, commands
    return matches[0]


def test_dry_run_default_mode(tmp_path):
    commands = _dry_run_commands(tmp_path, plan=DEFAULT_PLAN)
    assert _names(commands) == [
        "instagraal-pre",
        "instagraal",
        "instagraal-polish",
        "instagraal-refine",
        "instagraal-post",
        "instagraal-stats",
    ]
    assert "--bomb" not in _command(commands, "instagraal")
    polish_cmd = _command(commands, "instagraal-polish")
    assert polish_cmd[polish_cmd.index("--mode") + 1] == "fasta"  # the MCMC scaffolds as they are
    refine_cmd = _command(commands, "instagraal-refine")
    assert refine_cmd[refine_cmd.index("--info-frags") + 1].endswith("polished/new_info_frags.txt")
    assert refine_cmd[refine_cmd.index("--extend") + 1] == "auto"


def test_dry_run_cpu_mode(tmp_path):
    commands = _dry_run_commands(tmp_path, plan=CPU_PLAN)
    assert _names(commands) == ["instagraal-pre", "instagraal-refine", "instagraal-post", "instagraal-stats"]
    refine_cmd = _command(commands, "instagraal-refine")
    assert "--info-frags" not in refine_cmd
    assert refine_cmd[refine_cmd.index("--extend") + 1] == "on"
    assert refine_cmd[refine_cmd.index("--cool") + 1].endswith("pre/yeast.cool")
    assert _command(commands, "instagraal-post")[2].endswith("refined/refined_info_frags.txt")


def test_dry_run_legacy_mode(tmp_path):
    commands = _dry_run_commands(tmp_path, plan=LEGACY_PLAN)
    assert _names(commands) == ["instagraal-pre", "instagraal", "instagraal-polish", "instagraal-post", "instagraal-stats"]
    assert "--bomb" in _command(commands, "instagraal")
    assert "--mode" not in _command(commands, "instagraal-polish")
    assert _command(commands, "instagraal-post")[2].endswith("polished/new_info_frags.txt")
    stats_cmd = _command(commands, "instagraal-stats")
    assert stats_cmd[stats_cmd.index("--labels") + 1] == "Input assembly,Polished assembly"


def test_dry_run_no_polish_writes_scaffolds_as_is(tmp_path):
    commands = _dry_run_commands(tmp_path, plan=Plan(mcmc=True, bomb=True), criterion="sequences")
    polish_cmd = _command(commands, "instagraal-polish")
    assert polish_cmd[polish_cmd.index("--mode") + 1] == "fasta"
    assert "--criterion" not in polish_cmd
    # the size filters and junction still apply, so post sees a matching assembly
    assert "--min-scaffold-size" in polish_cmd
    assert polish_cmd[polish_cmd.index("--junction") + 1] == "NNNNNN"
    post_cmd = _command(commands, "instagraal-post")
    assert post_cmd[post_cmd.index("--junction") + 1] == "6"
    stats_cmd = _command(commands, "instagraal-stats")
    assert stats_cmd[stats_cmd.index("--labels") + 1] == "Input assembly,Scaffolded assembly (not polished)"


def test_dry_run_no_bomb(tmp_path):
    mcmc_cmd = _command(_dry_run_commands(tmp_path, plan=Plan(mcmc=True, polish=True)), "instagraal")
    assert "--bomb" not in mcmc_cmd
    mcmc_cmd = _command(_dry_run_commands(tmp_path, plan=LEGACY_PLAN), "instagraal")
    assert "--bomb" in mcmc_cmd


@pytest.mark.parametrize("junction,expected", [("", "0"), ("NNNNNN", "6"), ("AATTCC" * 2, "12")])
def test_dry_run_post_junction_matches_polish(tmp_path, junction, expected):
    commands = _dry_run_commands(tmp_path, junction=junction)
    post_cmd = _command(commands, "instagraal-post")
    assert post_cmd[post_cmd.index("--junction") + 1] == expected
    # even an empty junction is passed on: instagraal-polish must not fall back to its own default
    polish_cmd = _command(commands, "instagraal-polish")
    assert polish_cmd[polish_cmd.index("--junction") + 1] == junction


@pytest.mark.parametrize("plan", [DEFAULT_PLAN, CPU_PLAN])
@pytest.mark.parametrize("junction,expected", [("NNNNNN", "6"), ("", "0")])
def test_dry_run_refine_uses_refined_output(tmp_path, plan, junction, expected):
    commands = _dry_run_commands(tmp_path, plan=plan, junction=junction)
    refine_cmd = _command(commands, "instagraal-refine")
    # passed even when empty, like for instagraal-polish: post is told len(junction)
    assert refine_cmd[refine_cmd.index("--junction") + 1] == junction
    post_cmd = _command(commands, "instagraal-post")
    assert post_cmd[2].endswith("refined/refined_info_frags.txt")
    assert post_cmd[post_cmd.index("--junction") + 1] == expected
    stats_cmd = _command(commands, "instagraal-stats")
    assert stats_cmd[2].endswith("refined/refined_genome.fa")
    assert stats_cmd[stats_cmd.index("--labels") + 1] == "Input assembly,Refined assembly"


def test_dry_run_mcmc_polish_refine_refines_the_polished_layout(tmp_path):
    commands = _dry_run_commands(tmp_path, plan=Plan(mcmc=True, polish=True, refine=True))
    assert "--mode" not in _command(commands, "instagraal-polish")
    refine_cmd = _command(commands, "instagraal-refine")
    assert refine_cmd[refine_cmd.index("--info-frags") + 1].endswith("polished/new_info_frags.txt")


def test_dry_run_copies_no_final_assembly(tmp_path):
    _dry_run_commands(tmp_path, plan=CPU_PLAN)
    assert not (tmp_path / "final_genome.fa").exists()


def test_install_final_copies_the_assembly_to_fixed_names(tmp_path):
    (tmp_path / "refined").mkdir()
    fa, frags = tmp_path / "refined" / "refined_genome.fa", tmp_path / "refined" / "refined_info_frags.txt"
    fa.write_text(">refined_scaffold_1\nACGT\n")
    frags.write_text(">refined_scaffold_1\ninit_contig\tid_frag\torientation\tstart\tend\nctg\t0\t1\t0\t4\n")
    _install_final(fa, frags, tmp_path, dry_run=False)
    assert (tmp_path / "final_genome.fa").read_text() == fa.read_text()
    assert (tmp_path / "final_info_frags.txt").read_text() == frags.read_text()
