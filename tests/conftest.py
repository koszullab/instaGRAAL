"""Shared pytest configuration and session-scoped fixtures.

This file is loaded by pytest before any test module is imported, which makes
it the right place for:
  - global test-session setup (e.g. cleaning stale artefacts)
  - the matplotlib backend override (must happen before any import of
    matplotlib.pyplot further down the collection chain)
  - session-scoped fixtures shared between test modules
"""

import pathlib

import matplotlib
import pytest
from click.testing import CliRunner

from instagraal.cli.post import main as post_main
from instagraal.cli.pre import main as pre_main

# Force a non-interactive backend before any test file imports matplotlib.pyplot
matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Skip GPU-dependent tests (overrides auto-detection)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--no-gpu"):
        skip = pytest.mark.skip(reason="--no-gpu flag provided")
        for item in items:
            if item.fspath.basename == "test_instagraal_gpu.py":
                item.add_marker(skip)


REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"

REF_FASTA = TEST_DATA / "yeast.contigs.fa.gz"
REF_PAIRS = TEST_DATA / "yeast.pairs.gz"
NEW_INFO_FRAGS = TEST_DATA / "new_info_frags.txt"
ENZYMES = "DpnII,HinfI"


@pytest.fixture(scope="session")
def pre_output_dir(tmp_path_factory):
    """Run ``instagraal-pre`` on the example data and return the output dir.

    Shared by both the non-GPU tests (which validate the pre-processing
    outputs) and the GPU tests (which feed the pre-processing output directly
    into ``instagraal`` to exercise the full pipeline).
    """
    out = tmp_path_factory.mktemp("pre_out")
    runner = CliRunner()
    result = runner.invoke(
        pre_main,
        [
            str(REF_FASTA),
            str(REF_PAIRS),
            "--enzyme",
            ENZYMES,
            "--output-dir",
            str(out),
        ],
    )
    assert result.exit_code == 0, f"instagraal-pre failed (exit {result.exit_code}):\n{result.output}"
    return out


@pytest.fixture(scope="session")
def post_output_dir(tmp_path_factory):
    """Run ``instagraal-post`` on the test pairs + committed new_info_frags.

    Uses ``--no-balance`` so the test does not require a dense contact matrix.
    A low resolution (10 000 bp) is used to keep the mcool small.
    """
    out = tmp_path_factory.mktemp("post_out")
    runner = CliRunner()
    result = runner.invoke(
        post_main,
        [
            str(REF_PAIRS),
            str(NEW_INFO_FRAGS),
            "--resolutions",
            "10000",
            "--output-dir",
            str(out),
            "--no-balance",
        ],
    )
    assert result.exit_code == 0, f"instagraal-post failed (exit {result.exit_code}):\n{result.output}"
    return out
