"""instaGRAAL testing

Tests for the instagraal-pre preprocessing module, downstream compatibility
with the instaGRAAL scaffolder's pyramid-building step, and validation of
the instagraal main-command output artifacts.
"""

import pathlib
import shutil
import struct
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

REF_INFO_CONTIGS = EXAMPLE_DATA / "main" / "info_contigs.txt"
REF_FRAGMENTS_LIST = EXAMPLE_DATA / "main" / "fragments_list.txt"
REF_ABS_CONTACTS = EXAMPLE_DATA / "main" / "abs_fragments_contacts_weighted.txt"

# Expected values derived from the reference data set
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


# ---------------------------------------------------------------------------
# Tests: instagraal main-command output (pre-existing artefacts)
#
# The command was run as:
#   uv run instagraal example/data/main/ example/data/pre/metator_00056_00034.fa.gz
#       example/data/out/ --cycles 3 --bomb --save-matrix
#
# Because instagraal requires CUDA at import time the command cannot be
# re-invoked in CI.  Instead these tests validate the artefacts that were
# already produced and committed under example/data/out/.
# ---------------------------------------------------------------------------

# Configuration constants derived from the reference run
INSTAGRAAL_OUT = REPO_ROOT / "example" / "data" / "out"
MCMC_LEVEL = 4
MCMC_CYCLES = 3
MCMC_FRAGS = 825  # n_new_frags at level 4 for this data set (deterministic)
MCMC_N_ITERS = MCMC_CYCLES * MCMC_FRAGS  # lines written to list_* files

# instagraal is stochastic; the number of scaffolded contigs varies between
# runs.  The biologically meaningful check is that a reasonable number of
# large (>100 kb) assembly contigs are produced.
EXPECTED_LARGE_CONTIGS = 15  # expected contigs > 100 kb
LARGE_CONTIG_TOLERANCE = 3  # allowed deviation either side


@pytest.fixture(scope="module")
def instagraal_out_dir():
    """Return the committed instagraal output directory."""
    return INSTAGRAAL_OUT / "main" / f"test_mcmc_{MCMC_LEVEL}"


# ---------------------------------------------------------------------------
# Output directory structure
# ---------------------------------------------------------------------------


def test_instagraal_out_dir_exists(instagraal_out_dir):
    """The main output sub-directory exists."""
    assert instagraal_out_dir.is_dir()


@pytest.mark.parametrize(
    "fname",
    [
        "genome.fasta",
        "info_frags.txt",
        "list_likelihood.txt",
        "list_n_contigs.txt",
        "list_mean_len.txt",
        "list_dist_init_genome.txt",
        "list_mutations.txt",
        "save_simu_step_0.txt",
        f"save_simu_step_{MCMC_CYCLES - 1}.txt",
        "matrix_cycle_0.png",
        f"matrix_cycle_{MCMC_CYCLES - 1}.png",
    ],
)
def test_instagraal_expected_files_exist(instagraal_out_dir, fname):
    """All expected output files were created."""
    assert (instagraal_out_dir / fname).exists(), f"Missing: {fname}"


# ---------------------------------------------------------------------------
# genome.fasta
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def genome_fasta_lines(instagraal_out_dir):
    return (instagraal_out_dir / "genome.fasta").read_text().splitlines()


def test_genome_fasta_contig_count(genome_fasta_lines):
    """Scaffolded FASTA contains at least one contig."""
    headers = [l for l in genome_fasta_lines if l.startswith(">")]
    assert len(headers) >= 1


def test_genome_fasta_large_contig_count(genome_fasta_lines):
    """Between EXPECTED_LARGE_CONTIGS ± LARGE_CONTIG_TOLERANCE contigs exceed
    100 kb.  instagraal is stochastic so exact counts vary, but the assembly
    quality should remain in a biologically meaningful range."""
    seq_len = 0
    large = 0
    for line in genome_fasta_lines:
        if line.startswith(">"):
            if seq_len > 100_000:
                large += 1
            seq_len = 0
        else:
            seq_len += len(line)
    if seq_len > 100_000:  # last contig
        large += 1
    lo = EXPECTED_LARGE_CONTIGS - LARGE_CONTIG_TOLERANCE
    hi = EXPECTED_LARGE_CONTIGS + LARGE_CONTIG_TOLERANCE
    assert lo <= large <= hi, f"Expected {lo}–{hi} contigs >100 kb, got {large}"


def test_genome_fasta_contig_names(genome_fasta_lines):
    """All FASTA headers follow the '3C-assembly-contig_N' pattern."""
    headers = [l for l in genome_fasta_lines if l.startswith(">")]
    for h in headers:
        name = h.lstrip(">")
        assert name.startswith("3C-assembly-contig_"), f"Unexpected header: {h}"
        suffix = name.split("3C-assembly-contig_")[1]
        assert suffix.isdigit(), f"Non-numeric contig index in: {h}"


def test_genome_fasta_no_empty_sequences(genome_fasta_lines):
    """No contig in the scaffolded FASTA has an empty sequence."""
    current_header = None
    seq_started = False
    for line in genome_fasta_lines:
        if line.startswith(">"):
            if current_header is not None:
                assert seq_started, f"Empty sequence for {current_header}"
            current_header = line
            seq_started = False
        elif line.strip():
            seq_started = True
    if current_header is not None:
        assert seq_started, f"Empty sequence for {current_header}"


def test_genome_fasta_valid_bases(genome_fasta_lines):
    """Sequence lines contain only valid IUPAC nucleotide characters."""
    valid = set("ACGTNacgtn")
    for line in genome_fasta_lines:
        if not line.startswith(">"):
            assert set(line) <= valid, f"Invalid characters in sequence line: {line[:60]!r}"


# ---------------------------------------------------------------------------
# info_frags.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def info_frags_text(instagraal_out_dir):
    return (instagraal_out_dir / "info_frags.txt").read_text()


def test_info_frags_contig_count(info_frags_text):
    """info_frags.txt contains at least one contig block."""
    blocks = [b.strip() for b in info_frags_text.split(">") if b.strip()]
    assert len(blocks) >= 1


def test_info_frags_block_structure(info_frags_text):
    """Each block in info_frags.txt has a header line and a data section."""
    blocks = [b.strip() for b in info_frags_text.split(">") if b.strip()]
    expected_columns = {"init_contig", "id_frag", "orientation", "start", "end"}
    for block in blocks:
        lines = block.splitlines()
        assert len(lines) >= 2, f"Block too short: {lines}"
        col_header = lines[1].split()
        assert set(col_header) == expected_columns, f"Unexpected columns: {col_header}"
        # Every data row must have exactly 5 fields
        for row in lines[2:]:
            fields = row.split()
            assert len(fields) == 5, f"Unexpected field count in row: {row!r}"
        # Orientation must be +1 or -1
        for row in lines[2:]:
            ori = int(row.split()[2])
            assert ori in (1, -1), f"Invalid orientation value: {ori}"


# ---------------------------------------------------------------------------
# save_simu_step_*.txt
# ---------------------------------------------------------------------------


def test_save_simu_step_file_count(instagraal_out_dir):
    """Exactly one save_simu_step file exists per cycle."""
    files = sorted(instagraal_out_dir.glob("save_simu_step_*.txt"))
    assert len(files) == MCMC_CYCLES


def test_save_simu_step_row_count(instagraal_out_dir):
    """Each save_simu_step file has one row per fragment."""
    for i in range(MCMC_CYCLES):
        path = instagraal_out_dir / f"save_simu_step_{i}.txt"
        rows = path.read_text().splitlines()
        assert len(rows) == MCMC_FRAGS, f"Wrong row count in {path.name}: {len(rows)}"


def test_save_simu_step_column_format(instagraal_out_dir):
    """save_simu_step files have 4 numeric columns: pos, start_bp, id_c, ori."""
    path = instagraal_out_dir / "save_simu_step_0.txt"
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        assert len(fields) == 4, f"Line {lineno}: expected 4 fields, got {len(fields)}"
        pos, start_bp, id_c, ori = (int(f) for f in fields)
        assert ori in (1, -1), f"Line {lineno}: invalid orientation {ori}"


# ---------------------------------------------------------------------------
# list_*.txt files
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def list_likelihood(instagraal_out_dir):
    path = instagraal_out_dir / "list_likelihood.txt"
    return [float(l) for l in path.read_text().splitlines() if l.strip()]


@pytest.fixture(scope="module")
def list_n_contigs(instagraal_out_dir):
    path = instagraal_out_dir / "list_n_contigs.txt"
    return [int(l) for l in path.read_text().splitlines() if l.strip()]


def test_list_likelihood_length(list_likelihood):
    """list_likelihood.txt has one entry per MCMC iteration."""
    assert len(list_likelihood) == MCMC_N_ITERS


def test_list_likelihood_are_floats(list_likelihood):
    """All likelihood values are finite floats."""
    import math

    for v in list_likelihood:
        assert math.isfinite(v), f"Non-finite likelihood value: {v}"


def test_list_n_contigs_length(list_n_contigs):
    """list_n_contigs.txt has one entry per MCMC iteration."""
    assert len(list_n_contigs) == MCMC_N_ITERS


def test_list_n_contigs_positive(list_n_contigs):
    """All n_contigs values are positive integers."""
    for v in list_n_contigs:
        assert v > 0, f"Non-positive contig count: {v}"


def test_list_mean_len_length(instagraal_out_dir):
    """list_mean_len.txt has one entry per MCMC iteration."""
    path = instagraal_out_dir / "list_mean_len.txt"
    values = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(values) == MCMC_N_ITERS


def test_list_dist_init_genome_length(instagraal_out_dir):
    """list_dist_init_genome.txt has one entry per MCMC iteration."""
    path = instagraal_out_dir / "list_dist_init_genome.txt"
    values = [l for l in path.read_text().splitlines() if l.strip()]
    assert len(values) == MCMC_N_ITERS


# ---------------------------------------------------------------------------
# list_mutations.txt
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mutations_df(instagraal_out_dir):
    return pd.read_csv(
        instagraal_out_dir / "list_mutations.txt",
        sep="\t",
    )


def test_list_mutations_columns(mutations_df):
    """list_mutations.txt has the expected three columns."""
    assert list(mutations_df.columns) == ["id_fA", "id_fB", "id_mutation"]


def test_list_mutations_row_count(mutations_df):
    """list_mutations.txt has one data row per MCMC iteration."""
    assert len(mutations_df) == MCMC_N_ITERS


def test_list_mutations_fragment_ids_in_range(mutations_df):
    """Fragment IDs in list_mutations.txt are within the valid range."""
    assert mutations_df["id_fA"].between(0, MCMC_FRAGS - 1).all()
    assert mutations_df["id_fB"].between(0, MCMC_FRAGS - 1).all()


def test_list_mutations_op_codes_non_negative(mutations_df):
    """Mutation operation codes are non-negative integers."""
    assert (mutations_df["id_mutation"] >= 0).all()


# ---------------------------------------------------------------------------
# matrix_cycle_*.png  (--save-matrix artefacts)
# ---------------------------------------------------------------------------


def test_matrix_png_file_count(instagraal_out_dir):
    """Exactly one matrix PNG exists per cycle."""
    files = sorted(instagraal_out_dir.glob("matrix_cycle_*.png"))
    assert len(files) == MCMC_CYCLES


def test_matrix_pngs_are_valid(instagraal_out_dir):
    """Each matrix PNG has a valid PNG signature and non-zero dimensions."""
    _PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
    for i in range(MCMC_CYCLES):
        path = instagraal_out_dir / f"matrix_cycle_{i}.png"
        data = path.read_bytes()
        assert data[:8] == _PNG_SIGNATURE, f"{path.name} has invalid PNG signature"
        width = struct.unpack(">I", data[16:20])[0]
        height = struct.unpack(">I", data[20:24])[0]
        assert width > 0 and height > 0, f"{path.name} has zero dimensions"
