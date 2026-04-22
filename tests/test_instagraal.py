"""instaGRAAL testing

Tests for the instagraal-pre preprocessing module and downstream compatibility
with the instaGRAAL scaffolder's pyramid-building step.
"""

import pathlib
import shutil

import h5py
import pandas as pd
import pytest
from click.testing import CliRunner

from instagraal import pyramid_sparse
from instagraal.pre import main as pre_main

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
EXAMPLE_DATA = REPO_ROOT / "example" / "data"

REF_FASTA = EXAMPLE_DATA / "metator_00056_00034.fa.gz"
REF_PAIRS = EXAMPLE_DATA / "valid_idx_pcrfree.pairs.gz"
REF_INFO_CONTIGS = EXAMPLE_DATA / "info_contigs.txt"
REF_FRAGMENTS_LIST = EXAMPLE_DATA / "fragments_list.txt"
REF_ABS_CONTACTS = EXAMPLE_DATA / "abs_fragments_contacts_weighted.txt"

ENZYMES = "DpnII,HinfI"

# Expected values derived from the reference data set
EXPECTED_N_FRAGS = 94_744
EXPECTED_N_PIXELS = 1_220_986


# ---------------------------------------------------------------------------
# Session-scoped fixture: run instagraal-pre once for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pre_output_dir(tmp_path_factory):
    """Run ``instagraal-pre`` on the example data and return the output dir."""
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


# ---------------------------------------------------------------------------
# Tests: fragments_list.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fragments_out(pre_output_dir):
    return pd.read_csv(pre_output_dir / "fragments_list.txt", sep="\t")


@pytest.fixture(scope="session")
def fragments_ref():
    return pd.read_csv(REF_FRAGMENTS_LIST, sep="\t")


def test_fragment_count(fragments_out):
    """Number of restriction fragments matches the reference count."""
    assert len(fragments_out) == EXPECTED_N_FRAGS


def test_fragment_positions_match_reference(fragments_out, fragments_ref):
    """Fragment IDs and genomic coordinates (chrom, start, end, size) are
    identical to the reference produced by hicstuff.

    Note: ``gc_content`` differs because the hicstuff reference contains a
    known scaling artefact (~100× too small) and is excluded from comparison.
    """
    cols = ["id", "chrom", "start_pos", "end_pos", "size"]
    pd.testing.assert_frame_equal(
        fragments_out[cols].reset_index(drop=True),
        fragments_ref[cols].reset_index(drop=True),
        check_like=False,
    )


def test_gc_content_range(fragments_out):
    """GC content values are in [0, 1] and look biologically plausible."""
    gc = fragments_out["gc_content"]
    assert gc.between(0.0, 1.0).all(), "GC values outside [0, 1]"
    # Mean GC for a bacterial genome should be in a reasonable range
    assert 0.2 <= gc.mean() <= 0.8


# ---------------------------------------------------------------------------
# Tests: info_contigs.txt
# ---------------------------------------------------------------------------


def test_info_contigs_identical_to_reference(pre_output_dir):
    """info_contigs.txt is byte-for-byte identical to the reference file."""
    assert (pre_output_dir / "info_contigs.txt").read_bytes() == REF_INFO_CONTIGS.read_bytes()


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


@pytest.fixture(scope="session")
def pixels_ref():
    return pd.read_csv(
        REF_ABS_CONTACTS,
        sep="\t",
        skiprows=1,
        header=None,
        names=["bin1_id", "bin2_id", "count"],
    )


def test_pixel_count(pixels_out):
    """Number of non-zero contact pixels matches the reference."""
    assert len(pixels_out) == EXPECTED_N_PIXELS


def test_pixels_match_reference(pixels_out, pixels_ref):
    """Every (bin1_id, bin2_id, count) pixel entry matches the reference."""
    pd.testing.assert_frame_equal(
        pixels_out.reset_index(drop=True),
        pixels_ref.reset_index(drop=True),
    )


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
