"""Tests for the ``instagraal-endtoend`` CLI command.

All tests mock ``_run_endtoend`` so no GPU or heavy computation is required.
"""

import pathlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from instagraal.cli.endtoend import main

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"

REF_FASTA = TEST_DATA / "yeast.contigs.fa.gz"
REF_PAIRS = TEST_DATA / "yeast.pairs.gz"


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


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help():
    result = _invoke(["--help"])
    assert result.exit_code == 0
    assert "FASTA" in result.output
    assert "PAIRS" in result.output
    assert "--enzyme" in result.output


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
# Successful invocation (mocked pipeline)
# ---------------------------------------------------------------------------


def _mock_run_endtoend(*args, **kwargs) -> list[str]:
    return [
        "instagraal-pre ...",
        "instagraal ...",
        "instagraal-polish ...",
        "instagraal-post ...",
        "instagraal-stats ...",
    ]


@pytest.mark.skipif(not REF_FASTA.exists() or not REF_PAIRS.exists(), reason="Test data files not present")
def test_success_default_options(tmp_path):
    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_mock_run_endtoend):
        result = _invoke([*_base_args(), "--output-dir", tmp_path])
    assert result.exit_code == 0, result.output
    assert "ALL STEPS COMPLETED" in result.output


@pytest.mark.skipif(not REF_FASTA.exists() or not REF_PAIRS.exists(), reason="Test data files not present")
def test_recap_commands_printed(tmp_path):
    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_mock_run_endtoend):
        result = _invoke([*_base_args(), "--output-dir", tmp_path])
    assert "instagraal-pre" in result.output
    assert "instagraal-stats" in result.output


# ---------------------------------------------------------------------------
# Option forwarding
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not REF_FASTA.exists() or not REF_PAIRS.exists(), reason="Test data files not present")
@pytest.mark.parametrize(
    "extra_args,expected_kwargs",
    [
        (["--no-bomb"], {"bomb": False}),
        (["--no-polish"], {"polish": False}),
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


@pytest.mark.skipif(not REF_FASTA.exists() or not REF_PAIRS.exists(), reason="Test data files not present")
def test_defaults(tmp_path):
    captured: dict = {}

    def _capture(*args, **kwargs):
        captured.update(kwargs)
        return []

    with patch("instagraal.cli.endtoend._run_endtoend", side_effect=_capture):
        result = _invoke([*_base_args(), "--output-dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert captured["level"] == 4
    assert captured["cycles"] == 100
    assert captured["bomb"] is True
    assert captured["polish"] is True
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

    kwargs = dict(
        fasta=REF_FASTA,
        pairs=REF_PAIRS,
        enzymes=["DpnII"],
        output_dir=tmp_path,
        device=0,
        level=4,
        cycles=10,
        resolutions="1000",
        bomb=True,
        circular=False,
        save_matrix=True,
        save_pickle=False,
        simple=False,
        quiet=False,
        debug=False,
        balance=True,
        balance_args=None,
        coverage_std=1.0,
        neighborhood=5,
        min_scaffold_size=0,
        min_scaffold_length=0,
        junction="NNNNNN",
        criterion=None,
        cool_name=None,
        dry_run=True,
    )
    kwargs.update(overrides)
    return _run_endtoend(**kwargs)


def _command(commands: list[str], name: str) -> list[str]:
    import shlex

    matches = [shlex.split(c) for c in commands if pathlib.Path(shlex.split(c)[0]).name == name]
    assert len(matches) == 1, commands
    return matches[0]


def test_dry_run_polishes_by_default(tmp_path):
    polish_cmd = _command(_dry_run_commands(tmp_path), "instagraal-polish")
    assert "--mode" not in polish_cmd


def test_dry_run_no_polish_writes_scaffolds_as_is(tmp_path):
    commands = _dry_run_commands(tmp_path, polish=False, criterion="sequences")
    polish_cmd = _command(commands, "instagraal-polish")
    assert polish_cmd[polish_cmd.index("--mode") + 1] == "fasta"
    assert "--criterion" not in polish_cmd
    # the size filters and junction still apply, so post sees a matching assembly
    assert "--min-scaffold-size" in polish_cmd
    assert polish_cmd[polish_cmd.index("--junction") + 1] == "NNNNNN"
    post_cmd = _command(commands, "instagraal-post")
    assert post_cmd[post_cmd.index("--junction") + 1] == "6"


def test_dry_run_no_bomb(tmp_path):
    mcmc_cmd = _command(_dry_run_commands(tmp_path, bomb=False), "instagraal")
    assert "--bomb" not in mcmc_cmd
    mcmc_cmd = _command(_dry_run_commands(tmp_path, bomb=True), "instagraal")
    assert "--bomb" in mcmc_cmd


@pytest.mark.parametrize("junction,expected", [("", "0"), ("NNNNNN", "6"), ("AATTCC" * 2, "12")])
def test_dry_run_post_junction_matches_polish(tmp_path, junction, expected):
    commands = _dry_run_commands(tmp_path, junction=junction)
    post_cmd = _command(commands, "instagraal-post")
    assert post_cmd[post_cmd.index("--junction") + 1] == expected
    # even an empty junction is passed on: instagraal-polish must not fall back to its own default
    polish_cmd = _command(commands, "instagraal-polish")
    assert polish_cmd[polish_cmd.index("--junction") + 1] == junction
