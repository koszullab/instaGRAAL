"""Tests for the instagraal-polish CLI and the parse_info_frags module.

Unit tests use small synthetic scaffolds so they run without any on-disk files.
Integration tests use synthetic fixtures built on-the-fly from the shared
reference FASTA in tests/data/.
"""

import gzip
import pathlib

import pytest
from click.testing import CliRunner

from instagraal.cli.polish import main as polish_main
from instagraal.parse_info_frags import (
    _parse_fasta,
    correct_spurious_inversions,
    find_lost_dna,
    format_info_frags,
    integrate_lost_dna,
    parse_info_frags,
    rearrange_intra_scaffolds,
    reorient_consecutive_blocks,
    remove_spurious_insertions,
    write_fasta,
    write_info_frags,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).parent.parent
TEST_DATA = REPO_ROOT / "tests" / "data"
REF_FASTA_GZ = TEST_DATA / "yeast.contigs.fa.gz"


# ---------------------------------------------------------------------------
# Shared synthetic scaffolds
# ---------------------------------------------------------------------------

SCAFFOLDS_SIMPLE = {
    "scaffold1": [
        ["ctgA", 0, 0, 100, 1],
        ["ctgA", 1, 100, 200, 1],
        ["ctgA", 2, 200, 300, 1],
    ],
    "scaffold2": [
        ["ctgB", 0, 0, 150, 1],
        ["ctgB", 1, 150, 300, -1],
    ],
}

# Scaffold with a spurious singleton insertion
SCAFFOLDS_SINGLETON = {
    "scaffold1": [
        ["ctgA", 0, 0, 100, 1],
        ["ctgA", 1, 100, 200, 1],
        ["ctgX", 99, 5000, 5100, -1],  # singleton insertion
        ["ctgA", 2, 200, 300, 1],
        ["ctgA", 3, 300, 400, 1],
    ],
}

# Scaffold with a spurious inversion (colinear criterion)
SCAFFOLDS_INVERSION = {
    "scaffold1": [
        ["ctgA", 0, 0, 100, 1],
        ["ctgA", 1, 100, 200, 1],
        ["ctgA", 2, 200, 300, -1],  # inversion
        ["ctgA", 3, 300, 400, 1],
    ],
}

# Scaffold where one block appears out-of-order (rearrange target)
SCAFFOLDS_REARRANGE = {
    "scaffold1": [
        ["ctgA", 0, 0, 100, 1],
        ["ctgA", 1, 100, 200, 1],
        ["ctgB", 5, 500, 600, -1],
        ["ctgB", 6, 600, 700, -1],
        ["ctgA", 2, 200, 300, 1],  # second block of ctgA - should be moved
    ],
}


# ---------------------------------------------------------------------------
# Helper: tiny reference FASTA in-memory
# ---------------------------------------------------------------------------


def _make_fasta_file(tmp_path, records, compress=False):
    """Write records = {id: sequence_str} to a temp FASTA file."""
    lines = []
    for rec_id, seq in records.items():
        lines.append(f">{rec_id}")
        lines.append(seq)
    content = "\n".join(lines) + "\n"
    if compress:
        p = tmp_path / "ref.fa.gz"
        with gzip.open(p, "wt") as fh:
            fh.write(content)
    else:
        p = tmp_path / "ref.fa"
        p.write_text(content)
    return p


def _make_info_frags_file(tmp_path, scaffolds, filename="info_frags.txt"):
    """Serialise a scaffolds dict to a temporary info_frags.txt file using
    the canonical write_info_frags serialiser so the column order is correct.
    """
    p = tmp_path / filename
    write_info_frags(scaffolds, output=str(p))
    return p


# ===========================================================================
# Unit tests: parse_info_frags
# ===========================================================================


def test_parse_info_frags_round_trip(tmp_path):
    """parse_info_frags correctly reads back a written info_frags file."""
    p = _make_info_frags_file(tmp_path, SCAFFOLDS_SIMPLE)
    result = parse_info_frags(str(p))
    assert set(result.keys()) == {"scaffold1", "scaffold2"}
    assert len(result["scaffold1"]) == 3
    assert len(result["scaffold2"]) == 2
    assert result["scaffold1"][0] == ["ctgA", 0, 0, 100, 1]


def test_parse_info_frags_bin_format(tmp_path):
    """Each parsed bin has the expected [contig, id, start, end, ori] shape."""
    p = _make_info_frags_file(tmp_path, SCAFFOLDS_SIMPLE)
    result = parse_info_frags(str(p))
    for scaffold in result.values():
        for b in scaffold:
            contig, frag_id, start, end, ori = b
            assert isinstance(contig, str)
            assert isinstance(frag_id, int)
            assert start < end
            assert ori in {-1, 1}


# ===========================================================================
# Unit tests: format_info_frags
# ===========================================================================


def test_format_info_frags_passthrough_dict():
    """format_info_frags returns the dict unchanged when given a dict."""
    result = format_info_frags(SCAFFOLDS_SIMPLE)
    assert result is SCAFFOLDS_SIMPLE


def test_format_info_frags_reads_file(tmp_path):
    """format_info_frags can transparently load a file path."""
    p = _make_info_frags_file(tmp_path, SCAFFOLDS_SIMPLE)
    result = format_info_frags(str(p))
    assert set(result.keys()) == {"scaffold1", "scaffold2"}


# ===========================================================================
# Unit tests: remove_spurious_insertions
# ===========================================================================


def test_remove_spurious_insertions_removes_singleton():
    result = remove_spurious_insertions(SCAFFOLDS_SINGLETON)
    contigs = [b[0] for b in result["scaffold1"]]
    assert "ctgX" not in contigs


def test_remove_spurious_insertions_keeps_flanks():
    result = remove_spurious_insertions(SCAFFOLDS_SINGLETON)
    assert len(result["scaffold1"]) == 4


def test_remove_spurious_insertions_no_change_when_small():
    """Scaffolds with ≤ 2 bins are returned unchanged."""
    sc = {"s": [["ctgA", 0, 0, 100, 1], ["ctgB", 0, 0, 100, 1]]}
    result = remove_spurious_insertions(sc)
    assert result["s"] == sc["s"]


# ===========================================================================
# Unit tests: correct_spurious_inversions
# ===========================================================================


def test_correct_spurious_inversions_colinear():
    result = correct_spurious_inversions(SCAFFOLDS_INVERSION, criterion="colinear")
    orientations = [b[-1] for b in result["scaffold1"]]
    assert orientations == [1, 1, 1, 1]


def test_correct_spurious_inversions_cis():
    result = correct_spurious_inversions(SCAFFOLDS_INVERSION, criterion="cis")
    orientations = [b[-1] for b in result["scaffold1"]]
    assert orientations == [1, 1, 1, 1]


def test_correct_spurious_inversions_default_is_colinear():
    r_default = correct_spurious_inversions(SCAFFOLDS_INVERSION)
    r_colinear = correct_spurious_inversions(SCAFFOLDS_INVERSION, criterion="colinear")
    assert r_default == r_colinear


def test_correct_spurious_inversions_preserves_bins():
    result = correct_spurious_inversions(SCAFFOLDS_INVERSION)
    assert len(result["scaffold1"]) == len(SCAFFOLDS_INVERSION["scaffold1"])


# ===========================================================================
# Unit tests: reorient_consecutive_blocks
# ===========================================================================


def test_reorient_consecutive_blocks_blocks_mode():
    sc = {
        "scaffold1": [
            ["ctgA", 0, 0, 100, 1],
            ["ctgA", 1, 100, 200, -1],
            ["ctgA", 2, 200, 300, -1],
        ]
    }
    # Two negative, one positive → majority negative → all should be -1
    result = reorient_consecutive_blocks(sc, mode="blocks")
    orientations = [b[-1] for b in result["scaffold1"]]
    assert orientations == [-1, -1, -1]


def test_reorient_consecutive_blocks_preserves_bin_count():
    result = reorient_consecutive_blocks(SCAFFOLDS_SIMPLE, mode="blocks")
    for name in SCAFFOLDS_SIMPLE:
        assert len(result[name]) == len(SCAFFOLDS_SIMPLE[name])


# ===========================================================================
# Unit tests: rearrange_intra_scaffolds
# ===========================================================================


def test_rearrange_intra_scaffolds_groups_same_contig():
    result = rearrange_intra_scaffolds(SCAFFOLDS_REARRANGE)
    contigs = [b[0] for b in result["scaffold1"]]
    # All ctgA bins must be contiguous
    first_a = contigs.index("ctgA")
    last_a = len(contigs) - 1 - contigs[::-1].index("ctgA")
    for i in range(first_a, last_a + 1):
        assert contigs[i] == "ctgA"


def test_rearrange_intra_scaffolds_preserves_all_bins():
    result = rearrange_intra_scaffolds(SCAFFOLDS_REARRANGE)
    assert len(result["scaffold1"]) == len(SCAFFOLDS_REARRANGE["scaffold1"])


# ===========================================================================
# Unit tests: write_info_frags / round-trip
# ===========================================================================


def test_write_info_frags_round_trip(tmp_path):
    out = tmp_path / "out.txt"
    write_info_frags(SCAFFOLDS_SIMPLE, output=str(out))
    assert out.exists()
    result = parse_info_frags(str(out))
    assert set(result.keys()) == set(SCAFFOLDS_SIMPLE.keys())
    for name in SCAFFOLDS_SIMPLE:
        assert result[name] == SCAFFOLDS_SIMPLE[name]


# ===========================================================================
# Unit tests: _parse_fasta (gzip transparency)
# ===========================================================================


def test_parse_fasta_plain(tmp_path):
    records = {"seqA": "ACGTACGT", "seqB": "TTTTGGGG"}
    p = _make_fasta_file(tmp_path, records, compress=False)
    parsed = {r.id: str(r.seq) for r in _parse_fasta(p)}
    assert parsed == records


def test_parse_fasta_gzipped(tmp_path):
    records = {"seqA": "ACGTACGT", "seqB": "TTTTGGGG"}
    p = _make_fasta_file(tmp_path, records, compress=True)
    parsed = {r.id: str(r.seq) for r in _parse_fasta(p)}
    assert parsed == records


# ===========================================================================
# Unit tests: write_fasta
# ===========================================================================


def test_write_fasta_basic(tmp_path):
    """write_fasta produces a valid FASTA from a simple info_frags + reference."""
    ref = {"ctgA": "A" * 300, "ctgB": "C" * 300}
    fasta_file = _make_fasta_file(tmp_path, ref)
    info_file = _make_info_frags_file(tmp_path, SCAFFOLDS_SIMPLE)
    out_file = tmp_path / "out.fa"
    write_fasta(str(fasta_file), str(info_file), output=str(out_file))
    assert out_file.exists()
    content = out_file.read_text()
    assert ">scaffold1" in content
    assert ">scaffold2" in content


def test_write_fasta_sequence_length(tmp_path):
    """Sequences extracted by write_fasta have the expected total length."""
    ref = {"ctgA": "A" * 300, "ctgB": "C" * 300}
    fasta_file = _make_fasta_file(tmp_path, ref)
    info_file = _make_info_frags_file(tmp_path, SCAFFOLDS_SIMPLE)
    out_file = tmp_path / "out.fa"
    write_fasta(str(fasta_file), str(info_file), output=str(out_file))
    from Bio import SeqIO

    records = {r.id: r for r in SeqIO.parse(str(out_file), "fasta")}
    # scaffold1 covers ctgA[0:100] + ctgA[100:200] + ctgA[200:300] = 300 bp
    assert len(records["scaffold1"].seq) == 300
    # scaffold2 covers ctgB[0:150] + ctgB[150:300] = 300 bp (second bin rev-comp)
    assert len(records["scaffold2"].seq) == 300


def test_write_fasta_reverse_complement(tmp_path):
    """Bins with orientation -1 are reverse-complemented in the output."""
    ref = {"ctgA": "AAAA", "ctgB": "CCCC"}
    sc = {"s1": [["ctgA", 0, 0, 4, 1]], "s2": [["ctgB", 0, 0, 4, -1]]}
    fasta_file = _make_fasta_file(tmp_path, ref)
    info_file = _make_info_frags_file(tmp_path, sc)
    out_file = tmp_path / "out.fa"
    write_fasta(str(fasta_file), str(info_file), output=str(out_file))
    from Bio import SeqIO

    records = {r.id: str(r.seq) for r in SeqIO.parse(str(out_file), "fasta")}
    assert records["s1"] == "AAAA"
    assert records["s2"] == "GGGG"  # rev-comp of CCCC


def test_write_fasta_gzipped_input(tmp_path):
    """write_fasta accepts a gzip-compressed reference FASTA."""
    ref = {"ctgA": "ACGT" * 25}
    fasta_gz = _make_fasta_file(tmp_path, ref, compress=True)
    sc = {"s1": [["ctgA", 0, 0, 100, 1]]}
    info_file = _make_info_frags_file(tmp_path, sc)
    out_file = tmp_path / "out.fa"
    write_fasta(str(fasta_gz), str(info_file), output=str(out_file))
    assert out_file.exists()
    content = out_file.read_text()
    assert ">s1" in content


# ===========================================================================
# Unit tests: find_lost_dna / integrate_lost_dna
# ===========================================================================


def test_find_lost_dna_detects_gap(tmp_path):
    """find_lost_dna identifies bases not covered by any scaffold bin."""
    ref = {"ctgA": "A" * 200}
    fasta_file = _make_fasta_file(tmp_path, ref)
    # Only covers [0, 100) - the range [100, 200) is uncovered
    sc = {"s1": [["ctgA", 0, 0, 100, 1]]}
    lost = find_lost_dna(str(fasta_file), sc)
    assert "ctgA" in lost
    assert any(b[2] >= 100 for b in lost["ctgA"])


def test_find_lost_dna_no_gap(tmp_path):
    """find_lost_dna returns nothing for a fully covered contig."""
    ref = {"ctgA": "A" * 200}
    fasta_file = _make_fasta_file(tmp_path, ref)
    sc = {"s1": [["ctgA", 0, 0, 200, 1]]}
    lost = find_lost_dna(str(fasta_file), sc)
    assert "ctgA" not in lost


def test_integrate_lost_dna_adds_new_scaffold():
    """integrate_lost_dna appends uncovered contigs as new singleton scaffolds."""
    sc = {"s1": [["ctgA", 0, 0, 100, 1]]}
    lost = {"ctgB": [["ctgB", -1, 0, 50, 1]]}
    result = integrate_lost_dna(sc, lost)
    assert "ctgB" in result


# ===========================================================================
# Integration tests: CLI modes via CliRunner
# ===========================================================================


@pytest.fixture(scope="module")
def cli_runner():
    return CliRunner()


@pytest.fixture(scope="module")
def example_info_frags(tmp_path_factory):
    """Synthetic info_frags.txt consistent with example_ref_fasta."""
    p = tmp_path_factory.mktemp("polish_int")
    info_p = p / "info_frags.txt"
    write_info_frags(SCAFFOLDS_SIMPLE, output=str(info_p))
    return str(info_p)


@pytest.fixture(scope="module")
def example_ref_fasta(tmp_path_factory):
    """Synthetic reference FASTA whose contigs match example_info_frags."""
    p = tmp_path_factory.mktemp("polish_fasta")
    return str(_make_fasta_file(p, {"ctgA": "A" * 300, "ctgB": "C" * 300}))


def test_cli_fasta_mode(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'fasta' mode writes polished_genome.fa into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "fasta", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert (out_dir / "polished_genome.fa").exists()
    assert (out_dir / "polished_genome.fa").stat().st_size > 0


def test_cli_fasta_mode_produces_valid_fasta(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'fasta' mode output contains FASTA headers and sequences."""
    out_dir = tmp_path / "out"
    cli_runner.invoke(
        polish_main,
        ["-m", "fasta", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
    )
    from Bio import SeqIO

    records = list(SeqIO.parse(str(out_dir / "polished_genome.fa"), "fasta"))
    assert len(records) > 0
    for r in records:
        assert len(r.seq) > 0


def test_cli_singleton_mode(cli_runner, example_info_frags, tmp_path):
    """'singleton' mode writes new_info_frags.txt into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "singleton", "-i", example_info_frags, "-o", str(out_dir)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert (out_dir / "new_info_frags.txt").exists()
    sc = parse_info_frags(str(out_dir / "new_info_frags.txt"))
    assert len(sc) > 0


def test_cli_inversion_mode(cli_runner, example_info_frags, tmp_path):
    """'inversion' mode writes new_info_frags.txt into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "inversion", "-i", example_info_frags, "-o", str(out_dir)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert parse_info_frags(str(out_dir / "new_info_frags.txt"))


def test_cli_inversion2_mode(cli_runner, example_info_frags, tmp_path):
    """'inversion2' mode writes new_info_frags.txt into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "inversion2", "-i", example_info_frags, "-o", str(out_dir)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert parse_info_frags(str(out_dir / "new_info_frags.txt"))


def test_cli_rearrange_mode(cli_runner, example_info_frags, tmp_path):
    """'rearrange' mode writes new_info_frags.txt into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "rearrange", "-i", example_info_frags, "-o", str(out_dir)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert parse_info_frags(str(out_dir / "new_info_frags.txt"))


def test_cli_reincorporation_mode(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'reincorporation' mode writes new_info_frags.txt into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "reincorporation", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert parse_info_frags(str(out_dir / "new_info_frags.txt"))


def test_cli_polishing_mode(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'polishing' mode writes polished_genome.fa and new_info_frags.txt into the output directory."""
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert (out_dir / "polished_genome.fa").exists()
    assert (out_dir / "polished_genome.fa").stat().st_size > 0
    assert (out_dir / "new_info_frags.txt").exists()


def test_cli_polishing_mode_creates_missing_directory(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'polishing' mode creates the output directory when it does not exist."""
    out_dir = tmp_path / "nested" / "output"
    assert not out_dir.exists()
    result = cli_runner.invoke(
        polish_main,
        ["-m", "polishing", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    assert out_dir.is_dir()
    assert (out_dir / "polished_genome.fa").exists()


def test_cli_polishing_mode_fasta_valid(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'polishing' output is a valid multi-record FASTA with non-empty sequences."""
    out_dir = tmp_path / "out"
    cli_runner.invoke(
        polish_main,
        ["-m", "polishing", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
    )
    from Bio import SeqIO

    records = list(SeqIO.parse(str(out_dir / "polished_genome.fa"), "fasta"))
    assert len(records) > 0
    for r in records:
        assert len(r.seq) > 0, f"Empty sequence for {r.id}"


def test_cli_fasta_mode_requires_fasta(cli_runner, example_info_frags, tmp_path):
    """'fasta' mode raises an error when --fasta is not supplied."""
    result = cli_runner.invoke(
        polish_main,
        ["-m", "fasta", "-i", example_info_frags, "-o", str(tmp_path / "out")],
    )
    assert result.exit_code != 0


def test_cli_polishing_mode_requires_fasta(cli_runner, example_info_frags, tmp_path):
    """'polishing' mode raises an error when --fasta is not supplied."""
    result = cli_runner.invoke(
        polish_main,
        ["-m", "polishing", "-i", example_info_frags, "-o", str(tmp_path / "out")],
    )
    assert result.exit_code != 0


# ===========================================================================
# Integration tests: assembly stats are printed in FASTA-producing modes
# ===========================================================================


def test_cli_fasta_mode_prints_assembly_stats(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'fasta' mode calls print_assembly_stats on the produced FASTA."""
    from unittest.mock import patch

    out_dir = tmp_path / "out"
    with patch("instagraal.cli.polish.print_assembly_stats") as mock_stats:
        result = cli_runner.invoke(
            polish_main,
            ["-m", "fasta", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    mock_stats.assert_called_once()
    args, kwargs = mock_stats.call_args
    called_path = args[0]
    called_label = kwargs.get("label", args[1] if len(args) > 1 else "")
    assert "polished_genome.fa" in called_path
    assert "fasta" in called_label.lower()


def test_cli_polishing_mode_prints_assembly_stats(cli_runner, example_info_frags, example_ref_fasta, tmp_path):
    """'polishing' mode calls print_assembly_stats on the produced FASTA."""
    from unittest.mock import patch

    out_dir = tmp_path / "out"
    with patch("instagraal.cli.polish.print_assembly_stats") as mock_stats:
        result = cli_runner.invoke(
            polish_main,
            ["-m", "polishing", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
            catch_exceptions=False,
        )
    assert result.exit_code == 0, f"CLI failed:\n{result.output}"
    mock_stats.assert_called_once()
    args, kwargs = mock_stats.call_args
    called_path = args[0]
    called_label = kwargs.get("label", args[1] if len(args) > 1 else "")
    assert "polished_genome.fa" in called_path
    assert "polishing" in called_label.lower()


# ===========================================================================
# Lost DNA: exact half-open regions, each put back exactly once
# ===========================================================================


def test_find_lost_dna_exact_regions(tmp_path):
    """Lost regions are the exact complement of the [start, end) bins."""
    fasta_file = _make_fasta_file(tmp_path, {"ctgA": "A" * 1000, "ctgB": "C" * 50})
    sc = {"s1": [["ctgA", 0, 100, 300, 1], ["ctgA", 2, 500, 900, -1]]}
    lost = find_lost_dna(str(fasta_file), sc)
    assert lost["ctgA"] == [
        ["ctgA", -1, 0, 100, 1],
        ["ctgA", -1, 300, 500, 1],
        ["ctgA", -1, 900, 1000, 1],
    ]
    assert lost["ctgB"] == [["ctgB", -1, 0, 50, 1]]


def test_find_lost_dna_one_base_tail(tmp_path):
    """A single uncovered base at the end of a contig is found."""
    fasta_file = _make_fasta_file(tmp_path, {"ctgA": "A" * 101})
    lost = find_lost_dna(str(fasta_file), {"s1": [["ctgA", 0, 0, 100, 1]]})
    assert lost["ctgA"] == [["ctgA", -1, 100, 101, 1]]


@pytest.mark.parametrize(
    "ori,expected",
    [
        (1, [["ctgA", -1, 0, 100, 1], ["ctgA", 0, 100, 200, 1], ["ctgA", -1, 200, 300, 1]]),
        (-1, [["ctgA", -1, 200, 300, -1], ["ctgA", 0, 100, 200, -1], ["ctgA", -1, 0, 100, -1]]),
    ],
)
def test_integrate_lost_dna_placement_follows_orientation(ori, expected):
    """Regions go upstream/downstream of their neighbour bin, in its orientation."""
    sc = {"s1": [["ctgA", 0, 100, 200, ori]]}
    lost = {"ctgA": [["ctgA", -1, 0, 100, 1], ["ctgA", -1, 200, 300, 1]]}
    assert integrate_lost_dna(sc, lost) == {"s1": expected}


def test_integrate_lost_dna_first_bin_and_neighbours_untouched():
    """A region attached before the first bin does not move other bins."""
    sc = {"s1": [["ctgA", 0, 100, 200, 1], ["ctgB", 0, 0, 50, 1], ["ctgC", 0, 0, 70, -1]]}
    lost = {"ctgA": [["ctgA", -1, 0, 100, 1]]}
    assert integrate_lost_dna(sc, lost)["s1"] == [
        ["ctgA", -1, 0, 100, 1],
        ["ctgA", 0, 100, 200, 1],
        ["ctgB", 0, 0, 50, 1],
        ["ctgC", 0, 0, 70, -1],
    ]


def test_integrate_lost_dna_region_between_two_bins_used_once():
    """A region flanked by two bins (here in two scaffolds) is put back once."""
    sc = {"s1": [["ctgA", 0, 0, 100, 1]], "s2": [["ctgA", 2, 200, 300, 1]]}
    lost = {"ctgA": [["ctgA", -1, 100, 200, 1]]}
    result = integrate_lost_dna(sc, lost)
    all_bins = [b for s in result.values() for b in s]
    assert sum(1 for b in all_bins if b[2:4] == [100, 200]) == 1
    assert result["s1"] == [["ctgA", 0, 0, 100, 1], ["ctgA", -1, 100, 200, 1]]


def test_integrate_lost_dna_keeps_every_leftover_region():
    """Regions next to no bin all become scaffolds, not only the last one."""
    sc = {"s1": [["ctgA", 0, 100, 200, 1]]}
    lost = {
        "ctgA": [["ctgA", -1, 300, 400, 1], ["ctgA", -1, 500, 600, 1]],
        "ctgB": [["ctgB", -1, 0, 80, 1]],
    }
    result = integrate_lost_dna(sc, lost)
    assert result["ctgA_300_400"] == [["ctgA", -1, 300, 400, 1]]
    assert result["ctgA_500_600"] == [["ctgA", -1, 500, 600, 1]]
    assert result["ctgB"] == [["ctgB", -1, 0, 80, 1]]


def test_reorient_consecutive_blocks_unknown_mode():
    with pytest.raises(ValueError, match="Unknown reorientation mode"):
        reorient_consecutive_blocks(SCAFFOLDS_SIMPLE, mode="colinear")


def test_reorient_consecutive_blocks_does_not_mutate_input():
    sc = {"s": [["ctgA", 0, 0, 100, 1], ["ctgA", 1, 100, 200, -1], ["ctgA", 2, 200, 300, -1]]}
    reorient_consecutive_blocks(sc)
    assert [b[-1] for b in sc["s"]] == [1, -1, -1]


def test_correct_spurious_inversions_unknown_criterion():
    with pytest.raises(ValueError, match="Unknown criterion"):
        correct_spurious_inversions(SCAFFOLDS_INVERSION, criterion="blocks")


def test_parse_info_frags_without_trailing_newline(tmp_path):
    p = tmp_path / "info_frags.txt"
    p.write_text(">sc1\ninit_contig\tid_frag\torientation\tstart\tend\nctgA\t0\t1\t0\t100")
    assert parse_info_frags(str(p)) == {"sc1": [["ctgA", 0, 0, 100, 1]]}


def test_write_fasta_headers_are_clean(tmp_path):
    """FASTA headers are exactly the scaffold names (no trailing whitespace)."""
    fasta_file = _make_fasta_file(tmp_path, {"ctgA": "A" * 300, "ctgB": "C" * 300})
    info_file = _make_info_frags_file(tmp_path, {"3C-assembly|contig_1": [["ctgA", 0, 0, 100, 1]]})
    out_file = tmp_path / "out.fa"
    write_fasta(str(fasta_file), str(info_file), output=str(out_file))
    assert out_file.read_text().splitlines()[0] == ">3C-assembly|contig_1"


def test_write_fasta_accepts_scaffold_dict(tmp_path):
    fasta_file = _make_fasta_file(tmp_path, {"ctgA": "AAAACCCC", "ctgB": "GGGG"})
    sc = {"s1": [["ctgA", 0, 4, 8, -1], ["ctgB", 0, 0, 4, 1]]}
    out_file = tmp_path / "out.fa"
    write_fasta(str(fasta_file), sc, output=str(out_file), junction="NN")
    from Bio import SeqIO

    records = {r.id: str(r.seq) for r in SeqIO.parse(str(out_file), "fasta")}
    assert records == {"s1": "GGGGNNGGGG"}


def test_plot_contig_composition_no_scaffold(tmp_path):
    """The composition plot does not crash on an empty assembly."""
    from instagraal.parse_info_frags import plot_contig_composition

    info_file = _make_info_frags_file(tmp_path, {})
    plot_contig_composition(str(info_file), output_path=tmp_path / "plot.png")
    assert (tmp_path / "plot.png").exists()


def test_cli_fasta_mode_filters_and_writes_info_frags(cli_runner, example_ref_fasta, tmp_path):
    """'fasta' mode applies --min-scaffold-size and writes a matching new_info_frags.txt."""
    info_file = _make_info_frags_file(tmp_path, SCAFFOLDS_SIMPLE)
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-m", "fasta", "-i", str(info_file), "-f", example_ref_fasta, "-o", str(out_dir), "-s", "2"],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    from Bio import SeqIO

    records = [r.id for r in SeqIO.parse(str(out_dir / "polished_genome.fa"), "fasta")]
    assert records == ["scaffold1"]  # scaffold2 has 2 bins
    assert parse_info_frags(str(out_dir / "new_info_frags.txt")) == {"scaffold1": SCAFFOLDS_SIMPLE["scaffold1"]}


@pytest.mark.parametrize(
    "mode,criterion",
    [("polishing", "colinear"), ("inversion2", "cis"), ("inversion", "blocks")],
)
def test_cli_rejects_criterion_of_another_mode(cli_runner, example_info_frags, example_ref_fasta, tmp_path, mode, criterion):
    result = cli_runner.invoke(
        polish_main,
        ["-m", mode, "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(tmp_path / "out"), "-c", criterion],
    )
    assert result.exit_code == 2
    assert "--criterion" in result.output


# ===========================================================================
# Property test: polishing keeps every input base exactly once
# ===========================================================================


def _random_assembly(seed, tmp_path):
    """Random reference, a bin tiling of it, and an instaGRAAL-like scaffolding.

    As in instaGRAAL output, bins tile each contig from its start but may
    stop before its end, some contigs have no bin at all, and bins are
    shuffled, flipped and joined across contigs.
    """
    import itertools
    import random

    rng = random.Random(seed)
    ref = {}
    runs = []
    for c in range(12):
        name = f"ctg{c}"
        length = rng.randint(50, 3000)
        ref[name] = "".join(rng.choice("ACGT") for _ in range(length))
        if rng.random() < 0.15:
            continue  # contig entirely discarded (e.g. low coverage)
        end_covered = length if rng.random() < 0.5 else rng.randint(length // 2, length)
        cuts = sorted(rng.sample(range(1, end_covered), min(rng.randint(0, 8), end_covered - 1)))
        bounds = [0, *cuts, end_covered]
        bins = [[name, i, s, e, 1] for i, (s, e) in enumerate(itertools.pairwise(bounds))]
        i = 0
        while i < len(bins):
            n = rng.randint(1, 3)
            ori = rng.choice([1, -1])
            run = [[c_, id_, s, e, ori] for c_, id_, s, e, _ in bins[i : i + n]]
            runs.append(run[::-1] if ori == -1 else run)
            i += n
    rng.shuffle(runs)
    scaffolds = {}
    while runs:
        n = rng.randint(1, 4)
        scaffolds[f"3C-assembly|contig_{len(scaffolds)}"] = [b for r in runs[:n] for b in r]
        runs = runs[n:]
    for sc in scaffolds.values():  # a few spurious single-bin inversions
        for b in sc:
            if rng.random() < 0.1:
                b[-1] = -b[-1]
    fasta_file = _make_fasta_file(tmp_path, ref)
    info_file = _make_info_frags_file(tmp_path, scaffolds)
    return ref, fasta_file, info_file


def _assert_every_base_once(ref, out_dir, junction):
    from Bio import SeqIO
    from Bio.Seq import Seq

    from instagraal.post import _build_new_bins

    scaffolds = parse_info_frags(str(out_dir / "new_info_frags.txt"))
    seen = {name: [0] * len(seq) for name, seq in ref.items()}
    for sc in scaffolds.values():
        for contig, _, start, end, _ in sc:
            assert 0 <= start < end <= len(ref[contig]), (contig, start, end)
            for pos in range(start, end):
                seen[contig][pos] += 1
    for contig, counts in seen.items():
        assert set(counts) == {1}, f"{contig}: bases missing or duplicated"

    records = {r.id: str(r.seq) for r in SeqIO.parse(str(out_dir / "polished_genome.fa"), "fasta")}
    assert set(records) == set(scaffolds)
    for name, sc in scaffolds.items():
        pieces, previous = [], None
        for contig, _, start, end, ori in sc:
            if junction and previous not in (None, contig):
                pieces.append(junction)
            piece = ref[contig][start:end]
            pieces.append(piece if ori == 1 else str(Seq(piece).reverse_complement()))
            previous = contig
        assert records[name] == "".join(pieces)

    # instagraal-post derives scaffold lengths from new_info_frags.txt
    bins = _build_new_bins(scaffolds, junction_len=len(junction))
    post_lengths = bins.groupby("chrom")["end"].max().to_dict()
    assert post_lengths == {name: len(seq) for name, seq in records.items()}


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("min_size", [0, 1, 2])
def test_cli_polishing_keeps_every_base_once(cli_runner, tmp_path, seed, min_size):
    ref, fasta_file, info_file = _random_assembly(seed, tmp_path)
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-i", str(info_file), "-f", str(fasta_file), "-o", str(out_dir), "-j", "NNNNNN", "-s", str(min_size)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    _assert_every_base_once(ref, out_dir, "NNNNNN")


# ===========================================================================
# CLI defaults: instagraal-polish and instagraal-post agree on the junction
# ===========================================================================


def test_cli_polishing_default_junction_matches_post_default(cli_runner, tmp_path):
    """Run on their defaults (README steps 3 and 4), instagraal-polish and
    instagraal-post agree on the polished coordinates: the scaffold lengths
    that post rebuilds from new_info_frags.txt are the FASTA record lengths."""
    from Bio import SeqIO

    from instagraal.cli.post import main as post_main
    from instagraal.post import _build_new_bins

    (post_junction,) = [p.default for p in post_main.params if p.name == "junction"]
    ref = {"ctgA": "ACGTTGCA" * 40, "ctgB": "GATTACA" * 30, "ctgC": "CCGGA" * 30, "ctgD": "TTAGGC" * 15}
    # ctgA, reversed ctgB and ctgC are joined into one scaffold (two junctions);
    # ctgD was left out by the scaffolder and is put back as its own scaffold.
    scaffolds = {
        "3C-assembly|contig_0": [
            ["ctgA", 0, 0, 120, 1],
            ["ctgA", 1, 120, 320, 1],
            ["ctgB", 1, 90, 210, -1],
            ["ctgB", 0, 0, 90, -1],
            ["ctgC", 0, 0, 150, 1],
        ],
    }
    fasta_file = _make_fasta_file(tmp_path, ref)
    info_file = _make_info_frags_file(tmp_path, scaffolds)
    out_dir = tmp_path / "out"
    result = cli_runner.invoke(
        polish_main,
        ["-i", str(info_file), "-f", str(fasta_file), "-o", str(out_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    new_scaffolds = parse_info_frags(str(out_dir / "new_info_frags.txt"))
    # bins of different contigs are still joined, so the junction length matters
    assert any(len({b[0] for b in sc}) > 1 for sc in new_scaffolds.values())
    fasta_lengths = {r.id: len(r.seq) for r in SeqIO.parse(str(out_dir / "polished_genome.fa"), "fasta")}
    bins = _build_new_bins(new_scaffolds, junction_len=post_junction)
    assert bins.groupby("chrom")["end"].max().to_dict() == fasta_lengths
