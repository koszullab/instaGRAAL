"""instaGRAAL testing

Tests for the instagraal-pre preprocessing module and downstream compatibility
with the instaGRAAL scaffolder's pyramid-building step.

Validation of main-command (GPU) output artefacts lives in test_instagraal_gpu.py.
"""

import pathlib
import shutil
from unittest.mock import patch

import h5py
import pandas as pd
import pytest

from instagraal import pyramid_sparse

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXAMPLE_DATA = REPO_ROOT / "example" / "data"

# Expected values derived from the toy data set (deterministic for these inputs)
EXPECTED_N_FRAGS = 94_744
EXPECTED_N_PIXELS = 1_220_986

# pre_output_dir is a session-scoped fixture defined in conftest.py and shared
# with the GPU test suite.


# ---------------------------------------------------------------------------
# Tests: output file existence
# ---------------------------------------------------------------------------


def test_output_files_exist(pre_output_dir):
    """All expected output files are created."""
    for name in (
        "fragments_list.txt",
        "info_contigs.txt",
        "abs_fragments_contacts_weighted.txt",
        "valid_idx_pcrfree.cool",
    ):
        assert (pre_output_dir / name).exists(), f"Missing output: {name}"


def test_pre_logs_assembly_stats(tmp_path_factory):
    """instagraal-pre calls print_assembly_stats on the input FASTA."""
    from click.testing import CliRunner

    from instagraal.cli.pre import main as pre_main

    out = tmp_path_factory.mktemp("pre_stats_check")
    runner = CliRunner()
    with patch("instagraal.cli.pre.print_assembly_stats") as mock_stats:
        result = runner.invoke(
            pre_main,
            [
                str(EXAMPLE_DATA / "pre" / "metator_00056_00034.fa.gz"),
                str(EXAMPLE_DATA / "pre" / "valid_idx_pcrfree.pairs.gz"),
                "--enzyme",
                "DpnII,HinfI",
                "--output-dir",
                str(out),
            ],
        )
    assert result.exit_code == 0, f"instagraal-pre failed:\n{result.output}"
    mock_stats.assert_called_once()
    args, kwargs = mock_stats.call_args
    called_label = kwargs.get("label", args[1] if len(args) > 1 else "")
    assert "input" in called_label.lower()


# ---------------------------------------------------------------------------
# Tests: fragments_list.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fragments_out(pre_output_dir):
    return pd.read_csv(pre_output_dir / "fragments_list.txt", sep="\t")


def test_fragment_count(fragments_out):
    """Number of restriction fragments matches the expected count for the toy dataset."""
    assert len(fragments_out) == EXPECTED_N_FRAGS


def test_gc_content_range(fragments_out):
    """GC content values are in [0, 1] and look biologically plausible."""
    gc = fragments_out["gc_content"]
    assert gc.between(0.0, 1.0).all(), "GC values outside [0, 1]"
    # Mean GC for a bacterial genome should be in a reasonable range
    assert 0.2 <= gc.mean() <= 0.8


# ---------------------------------------------------------------------------
# Tests: abs_fragments_contacts_weighted.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pixels_out(pre_output_dir):
    return pd.read_csv(
        pre_output_dir / "abs_fragments_contacts_weighted.txt",
        sep="\t",
        skiprows=1,
        header=None,
        names=["bin1_id", "bin2_id", "count"],
    )


def test_pixel_count(pixels_out):
    """Number of non-zero contact pixels matches the expected count for the toy dataset."""
    assert len(pixels_out) == EXPECTED_N_PIXELS


# ---------------------------------------------------------------------------
# Tests: instaGRAAL pyramid build (CUDA-free parsing step)
# ---------------------------------------------------------------------------


def test_pyramid_build_from_pre_output(pre_output_dir, tmp_path):
    """pyramid_sparse.build() successfully parses the instagraal-pre output.

    This exercises the exact file-loading code path that instaGRAAL uses
    before any GPU operations, confirming that the output files are
    structurally compatible with the scaffolder.
    """
    # Copy the three text files into a fresh temp dir so the pyramid
    # sub-directories don't bleed into pre_output_dir across test runs.
    work = tmp_path / "hic_folder"
    work.mkdir()
    for name in (
        "fragments_list.txt",
        "info_contigs.txt",
        "abs_fragments_contacts_weighted.txt",
    ):
        shutil.copy(pre_output_dir / name, work / name)

    pyramid_sparse.build(str(work), size_pyramid=1, factor=3, min_bin_per_contig=1)

    hdf5_path = work / "pyramids" / "pyramid_1_no_thresh" / "pyramid.hdf5"
    assert hdf5_path.exists(), "pyramid.hdf5 was not created"

    with h5py.File(hdf5_path, "r") as f:
        assert f.attrs.get("0") == "done", "Pyramid level 0 not marked as done"
        assert "0" in f, "Pyramid level 0 dataset missing from HDF5"
