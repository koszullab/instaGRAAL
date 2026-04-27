"""Tests for the instagraal-post post-processing module.

Unit tests use small synthetic data and can run without the pairs file.
Integration tests use the session-scoped ``post_output_dir`` fixture from
conftest.py, which runs ``instagraal-post`` against the committed test data.
"""

import gzip
import pathlib

import cooler
import pytest

from instagraal.post import (
    _build_liftover_index,
    _build_new_bins,
    _pos_to_new_bin,
    _pos_to_new_coords,
    _scaffold_bins_from_extended,
    _fragment_pixels_to_scaffold_pixels,
    _binnify,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"
NEW_INFO_FRAGS = TEST_DATA / "new_info_frags.txt"
REF_PAIRS = TEST_DATA / "yeast.pairs.gz"


# ===========================================================================
# Unit tests - synthetic scaffolds (no files needed)
# ===========================================================================

# Minimal synthetic new_scaffolds dict that exercises all code paths:
#   scaffold_A : two forward fragments from the same contig
#   scaffold_B : one reverse fragment from contig_X
#   scaffold_C : two fragments from *different* contigs (junction test)
SYNTHETIC_SCAFFOLDS = {
    "scaffold_A": [
        ["ctgA", 0, 0, 100, 1],
        ["ctgA", 1, 100, 200, 1],
    ],
    "scaffold_B": [
        ["ctgX", 5, 500, 600, -1],
    ],
    "scaffold_C": [
        ["ctgA", 2, 200, 300, 1],
        ["ctgB", 0, 0, 80, 1],
    ],
}


# ---------------------------------------------------------------------------
# _build_new_bins
# ---------------------------------------------------------------------------


def test_build_new_bins_no_junction():
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    # Total rows = 2 + 1 + 2 = 5
    assert len(bins) == 5


def test_build_new_bins_columns():
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    for col in ("chrom", "start", "end", "_orig_chrom", "_orig_start", "_orig_end", "_orientation"):
        assert col in bins.columns


def test_build_new_bins_start_end_monotone():
    """Within each scaffold bins must be non-overlapping and monotone."""
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    for _chrom, grp in bins.groupby("chrom", sort=False):
        starts = grp["start"].to_numpy()
        ends = grp["end"].to_numpy()
        assert (ends > starts).all(), "end must be > start for every bin"
        # consecutive bins must be contiguous (no junction in this call)
        assert (starts[1:] == ends[:-1]).all(), "bins must be contiguous"


def test_build_new_bins_junction_inserted_between_diff_contigs():
    """Junction bases are added only when the source contig changes."""
    bins_no_junc = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    bins_junc = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=6)

    # scaffold_A: same contig throughout — junction must NOT be added
    a_no = bins_no_junc[bins_no_junc["chrom"] == "scaffold_A"]
    a_j = bins_junc[bins_junc["chrom"] == "scaffold_A"]
    assert a_no["end"].iloc[-1] == a_j["end"].iloc[-1], "same-contig scaffold must not grow"

    # scaffold_C: two different contigs — junction of 6 bp must be inserted
    c_no = bins_no_junc[bins_no_junc["chrom"] == "scaffold_C"]
    c_j = bins_junc[bins_junc["chrom"] == "scaffold_C"]
    total_no = c_no["end"].iloc[-1]
    total_j = c_j["end"].iloc[-1]
    assert total_j == total_no + 6, f"expected +6 from junction, got {total_j - total_no}"


def test_build_new_bins_reverse_frag_size_preserved():
    """A reverse-complement fragment must have the same length as the original."""
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    b_row = bins[bins["_orig_chrom"] == "ctgX"].iloc[0]
    orig_len = b_row["_orig_end"] - b_row["_orig_start"]
    new_len = b_row["end"] - b_row["start"]
    assert orig_len == new_len


# ---------------------------------------------------------------------------
# _build_liftover_index
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def liftover_index():
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    return _build_liftover_index(bins)


def test_liftover_index_has_all_orig_chroms(liftover_index):
    assert set(liftover_index.keys()) == {"ctgA", "ctgX", "ctgB"}


def test_liftover_index_arrays_same_length(liftover_index):
    for entry in liftover_index.values():
        n = len(entry["orig_starts"])
        for key in ("orig_ends", "bin_ids", "new_chroms", "new_starts", "new_ends", "orientations"):
            assert len(entry[key]) == n, f"length mismatch in '{key}'"


# ---------------------------------------------------------------------------
# _pos_to_new_bin
# ---------------------------------------------------------------------------


def test_pos_to_new_bin_forward_hit(liftover_index):
    # ctgA fragment 0 covers 0-100, position 50 (1-based) → bin 0
    result = _pos_to_new_bin("ctgA", 50, liftover_index)
    assert result is not None


def test_pos_to_new_bin_at_boundary(liftover_index):
    # position 1 (1-based) → start of ctgA frag 0
    r1 = _pos_to_new_bin("ctgA", 1, liftover_index)
    # position 100 (1-based, last base of frag 0 → 0-based 99, in [0,100))
    r2 = _pos_to_new_bin("ctgA", 100, liftover_index)
    assert r1 == r2, "positions within the same fragment must map to the same bin"


def test_pos_to_new_bin_returns_none_unknown_chrom(liftover_index):
    assert _pos_to_new_bin("nonexistent", 1, liftover_index) is None


def test_pos_to_new_bin_returns_none_out_of_range(liftover_index):
    # ctgA covers 0-300 total; position 999 is beyond all fragments
    assert _pos_to_new_bin("ctgA", 999, liftover_index) is None


# ---------------------------------------------------------------------------
# _pos_to_new_coords
# ---------------------------------------------------------------------------


def test_pos_to_new_coords_forward(liftover_index):
    # ctgA frag 0: orig [0,100), ori +1; pos 1 (1-based) → new_start + 0
    result = _pos_to_new_coords("ctgA", 1, liftover_index)
    assert result is not None
    new_chrom, new_pos = result
    assert new_chrom == "scaffold_A"
    assert new_pos >= 1


def test_pos_to_new_coords_reverse_complement(liftover_index):
    # ctgX frag 5: orig [500,600), ori -1.
    # pos = 501 (1-based) → 0-based offset 0 → mirrored to frag_len-1 = 99
    r_start = _pos_to_new_coords("ctgX", 501, liftover_index)
    r_end = _pos_to_new_coords("ctgX", 599, liftover_index)
    assert r_start is not None and r_end is not None
    # start of original → last position in new fragment
    # end of original   → first position in new fragment
    assert r_start[1] > r_end[1], "reverse complement: earlier orig pos → later new pos"


def test_pos_to_new_coords_returns_none_unknown_chrom(liftover_index):
    assert _pos_to_new_coords("ghost", 1, liftover_index) is None


# ---------------------------------------------------------------------------
# _scaffold_bins_from_extended
# ---------------------------------------------------------------------------


def test_scaffold_bins_one_row_per_scaffold():
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    scaffold_bins = _scaffold_bins_from_extended(bins)
    assert len(scaffold_bins) == len(SYNTHETIC_SCAFFOLDS)
    assert list(scaffold_bins["chrom"]) == list(SYNTHETIC_SCAFFOLDS.keys())


def test_scaffold_bins_span_whole_scaffold():
    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    scaffold_bins = _scaffold_bins_from_extended(bins)
    # scaffold_A: two fragments [0,100) [100,200) → end = 200
    row_a = scaffold_bins[scaffold_bins["chrom"] == "scaffold_A"].iloc[0]
    assert row_a["start"] == 0
    assert row_a["end"] == 200


# ---------------------------------------------------------------------------
# _fragment_pixels_to_scaffold_pixels
# ---------------------------------------------------------------------------


def test_fragment_pixels_to_scaffold_pixels_sums_correctly():
    import pandas as pd

    bins = _build_new_bins(SYNTHETIC_SCAFFOLDS, junction_len=0)
    scaffold_bins = _scaffold_bins_from_extended(bins)
    # Two pixels on scaffold_A (bins 0 and 1 both belong to scaffold_A)
    fragment_pixels = pd.DataFrame({"bin1_id": [0, 0], "bin2_id": [0, 1], "count": [3, 5]})
    sp = _fragment_pixels_to_scaffold_pixels(fragment_pixels, bins, scaffold_bins)
    # Both pixels map to scaffold_A (idx 0) self-contact
    assert len(sp) == 1
    assert int(sp["count"].iloc[0]) == 8


# ---------------------------------------------------------------------------
# _binnify
# ---------------------------------------------------------------------------


def test_binnify_single_chrom():
    bins = _binnify({"chrA": 250}, 100)
    assert len(bins) == 3  # [0,100), [100,200), [200,250)
    assert int(bins["end"].iloc[-1]) == 250


def test_binnify_preserves_chrom_order():
    bins = _binnify({"chrB": 100, "chrA": 100}, 100)
    assert list(bins["chrom"]) == ["chrB", "chrA"]


# ===========================================================================
# Integration tests - use committed test data via post_output_dir fixture
# ===========================================================================


def test_post_output_files_exist(post_output_dir):
    """All expected output files are produced."""
    stem = "yeast"
    for fname in (
        f"{stem}_contigs.cool",
        f"{stem}_scaffolds.cool",
        f"{stem}_scaffolds_binned.mcool",
        f"{stem}_lifted.pairs.gz",
        f"{stem}_contigs.png",
        f"{stem}_scaffolds.png",
        f"{stem}_ps_curves.png",
    ):
        assert (post_output_dir / fname).exists(), f"Missing output: {fname}"


# ---------------------------------------------------------------------------
# _contigs.cool
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contigs_cool(post_output_dir):
    return cooler.Cooler(str(post_output_dir / "yeast_contigs.cool"))


def test_contigs_cool_has_bins(contigs_cool):
    assert contigs_cool.info["nbins"] > 0


def test_contigs_cool_has_pixels(contigs_cool):
    assert contigs_cool.info["nnz"] > 0


def test_contigs_cool_bins_are_whole_contigs(contigs_cool):
    """Every bin should start at 0 (each bin = one entire contig)."""
    bins = contigs_cool.bins()[:]
    assert (bins["start"] == 0).all(), "all contig bins must start at 0"


def test_contigs_cool_pixels_upper_triangular(contigs_cool):
    pixels = contigs_cool.pixels()[:]
    assert (pixels["bin1_id"] <= pixels["bin2_id"]).all()


def test_contigs_cool_pixel_counts_positive(contigs_cool):
    pixels = contigs_cool.pixels()[:]
    assert (pixels["count"] > 0).all()


# ---------------------------------------------------------------------------
# _scaffolds.cool
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scaffolds_cool(post_output_dir):
    return cooler.Cooler(str(post_output_dir / "yeast_scaffolds.cool"))


def test_scaffolds_cool_has_bins(scaffolds_cool):
    assert scaffolds_cool.info["nbins"] > 0


def test_scaffolds_cool_one_bin_per_scaffold(scaffolds_cool):
    """Number of bins must equal number of chromosomes (one bin per scaffold)."""
    assert scaffolds_cool.info["nbins"] == scaffolds_cool.info["nchroms"]


def test_scaffolds_cool_bins_are_whole_scaffolds(scaffolds_cool):
    bins = scaffolds_cool.bins()[:]
    chroms = scaffolds_cool.chroms()[:]
    # Each scaffold should have exactly one bin spanning [0, scaffold_length]
    for _, row in chroms.iterrows():
        chrom_bins = bins[bins["chrom"] == row["name"]]
        assert len(chrom_bins) == 1, f"scaffold {row['name']} must have exactly 1 bin"
        assert int(chrom_bins["start"].iloc[0]) == 0
        assert int(chrom_bins["end"].iloc[0]) == int(row["length"])


def test_scaffolds_cool_pixels_upper_triangular(scaffolds_cool):
    pixels = scaffolds_cool.pixels()[:]
    assert (pixels["bin1_id"] <= pixels["bin2_id"]).all()


def test_scaffolds_cool_pixel_counts_positive(scaffolds_cool):
    pixels = scaffolds_cool.pixels()[:]
    assert (pixels["count"] > 0).all()


# ---------------------------------------------------------------------------
# mcool
# ---------------------------------------------------------------------------


def test_post_mcool_has_resolutions(post_output_dir):
    mcool_path = str(post_output_dir / "yeast_scaffolds_binned.mcool")
    resolutions = cooler.fileops.list_coolers(mcool_path)
    assert len(resolutions) >= 1, "mcool contains no resolutions"


def test_post_mcool_10000_readable(post_output_dir):
    uri = str(post_output_dir / "yeast_scaffolds_binned.mcool") + "::resolutions/10000"
    clr = cooler.Cooler(uri)
    assert clr.info["nbins"] > 0


def test_post_mcool_bins_are_fixed_size(post_output_dir):
    """Bins in the mcool (except the last on each chrom) must have uniform size."""
    uri = str(post_output_dir / "yeast_scaffolds_binned.mcool") + "::resolutions/10000"
    clr = cooler.Cooler(uri)
    bins = clr.bins()[:]
    for chrom, grp in bins.groupby("chrom", sort=False):
        sizes = (grp["end"] - grp["start"]).to_numpy()
        # All bins except the last must be exactly 10000 bp
        if len(sizes) > 1:
            assert (sizes[:-1] == 10000).all(), f"non-uniform bin sizes on {chrom}"


# ---------------------------------------------------------------------------
# remapped pairs file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lifted_pairs_lines(post_output_dir):
    with gzip.open(post_output_dir / "yeast_lifted.pairs.gz", "rt") as fh:
        return fh.read().splitlines()


def test_lifted_pairs_has_header(lifted_pairs_lines):
    assert any(line.startswith("##") for line in lifted_pairs_lines)


def test_lifted_pairs_has_columns_line(lifted_pairs_lines):
    assert any(line.startswith("#columns:") for line in lifted_pairs_lines)


def test_lifted_pairs_data_rows_have_new_chroms(lifted_pairs_lines):
    """Data rows must reference new-assembly scaffold names (not original contig names)."""
    from instagraal.parse_info_frags import parse_info_frags

    new_scaffolds = parse_info_frags(str(NEW_INFO_FRAGS))
    new_chrom_names = set(new_scaffolds.keys())

    col_chr1, col_chr2 = 1, 3  # standard 4DN defaults
    for line in lifted_pairs_lines:
        if line.startswith("#"):
            if line.startswith("#columns:"):
                cols = line.split()[1:]
                try:
                    col_chr1 = cols.index("chr1")
                    col_chr2 = cols.index("chr2")
                except ValueError:
                    pass
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        assert parts[col_chr1] in new_chrom_names, f"chr1 {parts[col_chr1]!r} not in new assembly"
        assert parts[col_chr2] in new_chrom_names, f"chr2 {parts[col_chr2]!r} not in new assembly"
        break  # checking one data row is sufficient for a smoke-test


def test_lifted_pairs_positions_positive(lifted_pairs_lines):
    col_pos1, col_pos2 = 2, 4
    for line in lifted_pairs_lines:
        if line.startswith("#"):
            if line.startswith("#columns:"):
                cols = line.split()[1:]
                try:
                    col_pos1 = cols.index("pos1")
                    col_pos2 = cols.index("pos2")
                except ValueError:
                    pass
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        assert int(parts[col_pos1]) > 0
        assert int(parts[col_pos2]) > 0
        break


# ---------------------------------------------------------------------------
# HiC map PNGs
# ---------------------------------------------------------------------------


def test_post_contigs_hic_map_png_valid(post_output_dir):
    """Contig-level HiC map PNG must have valid PNG magic bytes."""
    data = (post_output_dir / "yeast_contigs.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG file"


def test_post_scaffolds_hic_map_png_valid(post_output_dir):
    """Scaffold-level HiC map PNG must have valid PNG magic bytes."""
    data = (post_output_dir / "yeast_scaffolds.png").read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "Not a valid PNG file"


def test_post_hic_map_png_non_empty(post_output_dir):
    size = (post_output_dir / "yeast_scaffolds.png").stat().st_size
    assert size > 1000, "PNG suspiciously small"
