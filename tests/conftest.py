"""Shared pytest configuration and session-scoped fixtures.

This file is loaded by pytest before any test module is imported, which makes
it the right place for:
  - global test-session setup (e.g. cleaning stale artefacts)
  - the matplotlib backend override (must happen before any import of
    matplotlib.pyplot further down the collection chain)
  - session-scoped fixtures shared between test modules
"""

import pathlib
import shutil

import matplotlib
import pytest
from click.testing import CliRunner

from instagraal.cli.pre import main as pre_main

# Force a non-interactive backend before any test file imports matplotlib.pyplot
matplotlib.use("Agg")

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXAMPLE_DATA = REPO_ROOT / "example" / "data"
PYRAMIDS_DIR = EXAMPLE_DATA / "out" / "pyramids"

REF_FASTA = EXAMPLE_DATA / "pre" / "metator_00056_00034.fa.gz"
REF_PAIRS = EXAMPLE_DATA / "pre" / "valid_idx_pcrfree.pairs.gz"
ENZYMES = "DpnII,HinfI"


@pytest.fixture(scope="session", autouse=True)
def clean_example_pyramids():
    """Force-remove ``example/data/out/pyramids`` at the start of every test
    session.

    The ``instagraal`` main command writes its pyramid artefacts into this
    directory.  Removing it before tests ensures:

    * The GPU test suite always starts from a clean state and exercises the
      full pyramid-build code path.
    * Stale pyramids from a previous manual run never silently hide a
      regression.

    The directory is untracked (listed in ``.gitignore``), so deleting it is
    safe.
    """
    if PYRAMIDS_DIR.exists():
        shutil.rmtree(PYRAMIDS_DIR)
    yield


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
