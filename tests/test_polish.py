"""Tests for the instagraal-polish CLI and the parse_info_frags module.

Unit tests use small synthetic scaffolds so they run without any on-disk files.
Integration tests use the committed example data under example/data/out/ and
the reference FASTA under example/data/pre/.
"""

import gzip
import pathlib
import textwrap

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
EXAMPLE_DATA = REPO_ROOT / "example" / "data"
REF_FASTA_GZ = EXAMPLE_DATA / "pre" / "metator_00056_00034.fa.gz"
INSTAGRAAL_OUT = EXAMPLE_DATA / "out" / "main" / "test_mcmc_4"
INFO_FRAGS_FILE = INSTAGRAAL_OUT / "info_frags.txt"


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
        ["ctgA", 2, 200, 300, 1],  # second block of ctgA – should be moved
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
    # Only covers [0, 100) – the range [100, 200) is uncovered
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
def example_info_frags():
    return str(INFO_FRAGS_FILE)


@pytest.fixture(scope="module")
def example_ref_fasta():
    return str(REF_FASTA_GZ)


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
        ["-m", "polishing", "-i", example_info_frags, "-f", example_ref_fasta, "-o", str(out_dir)],
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
