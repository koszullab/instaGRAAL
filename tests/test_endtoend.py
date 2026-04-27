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
    assert captured["save_matrix"] is True
    assert captured["balance"] is True
    assert captured["device"] == 0
    assert captured["min_scaffold_size"] == 0
    assert captured["min_scaffold_length"] == 0
    assert captured["junction"] == "NNNNNN"
