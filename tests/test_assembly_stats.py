"""Tests for the assembly_stats module.

All tests use small, synthetic in-memory / temp-file FASTA inputs and do
not require any external databases or tools.
"""

import gzip
import pathlib

import pytest
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from instagraal.assembly_stats import (
    compute_assembly_stats,
    format_assembly_stats,
    format_comparison_table,
    log_assembly_stats,
    print_assembly_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_fasta(tmp_path: pathlib.Path, records: dict, gz: bool = False) -> pathlib.Path:
    """Write {id: sequence} records to a temporary FASTA file."""
    seq_records = [SeqRecord(Seq(seq), id=name, description="") for name, seq in records.items()]
    if gz:
        p = tmp_path / "seqs.fa.gz"
        with gzip.open(p, "wt") as fh:
            SeqIO.write(seq_records, fh, "fasta")
    else:
        p = tmp_path / "seqs.fa"
        SeqIO.write(seq_records, str(p), "fasta")
    return p


# ---------------------------------------------------------------------------
# compute_assembly_stats – basic counts
# ---------------------------------------------------------------------------


class TestComputeAssemblyStats:
    """Unit tests for compute_assembly_stats."""

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.fa"
        p.write_text("")
        stats = compute_assembly_stats(str(p))
        assert stats["n_contigs"] == 0
        assert stats["total_length"] == 0

    def test_n_contigs(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "ACGT" * 10, "b": "GCGC" * 5})
        stats = compute_assembly_stats(str(fa))
        assert stats["n_contigs"] == 2

    def test_total_length(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "C" * 200})
        stats = compute_assembly_stats(str(fa))
        assert stats["total_length"] == 300

    def test_largest_and_shortest(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "C" * 200, "c": "G" * 50})
        stats = compute_assembly_stats(str(fa))
        assert stats["largest_contig"] == 200
        assert stats["shortest_contig"] == 50

    def test_mean_length(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "C" * 300})
        stats = compute_assembly_stats(str(fa))
        assert stats["mean_length"] == pytest.approx(200.0)

    def test_median_length(self, tmp_path):
        # three sequences: 100, 200, 300 → median = 200
        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "C" * 200, "c": "G" * 300})
        stats = compute_assembly_stats(str(fa))
        assert stats["median_length"] == pytest.approx(200.0)

    def test_gc_content_all_gc(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "GCGCGCGC"})
        stats = compute_assembly_stats(str(fa))
        assert stats["gc_content"] == pytest.approx(1.0)

    def test_gc_content_all_at(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "ATATATAT"})
        stats = compute_assembly_stats(str(fa))
        assert stats["gc_content"] == pytest.approx(0.0)

    def test_gc_content_mixed(self, tmp_path):
        # ACGT → 2 GC out of 4 = 0.5
        fa = _write_fasta(tmp_path, {"a": "ACGT"})
        stats = compute_assembly_stats(str(fa))
        assert stats["gc_content"] == pytest.approx(0.5)

    def test_gzip_input(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "G" * 200}, gz=True)
        stats = compute_assembly_stats(str(fa))
        assert stats["n_contigs"] == 2
        assert stats["total_length"] == 300


# ---------------------------------------------------------------------------
# compute_assembly_stats – N50 / L50 / N90 / L90
# ---------------------------------------------------------------------------


class TestNxLxMetrics:
    """Verify N50/L50/N90/L90 against hand-calculated expectations."""

    def test_n50_simple(self, tmp_path):
        # sequences: 100, 200, 300, 400, 500  → total 1500, 50% = 750
        # sorted desc: 500, 400, 300, 200, 100
        # cumulative:  500, 900 → N50 = 400 (first contig bringing cumsum ≥ 750)
        fa = _write_fasta(
            tmp_path,
            {"a": "A" * 100, "b": "G" * 200, "c": "C" * 300, "d": "T" * 400, "e": "A" * 500},
        )
        stats = compute_assembly_stats(str(fa))
        assert stats["n50"] == 400

    def test_l50_simple(self, tmp_path):
        fa = _write_fasta(
            tmp_path,
            {"a": "A" * 100, "b": "B" * 200, "c": "C" * 300, "d": "D" * 400, "e": "E" * 500},
        )
        stats = compute_assembly_stats(str(fa))
        # two contigs (500, 400) cover 900/1500 ≥ 50%
        assert stats["l50"] == 2

    def test_n90_simple(self, tmp_path):
        # total 1500, 90% = 1350
        # sorted desc cumsum: 500, 900, 1200, 1400 → N90 = 200
        fa = _write_fasta(
            tmp_path,
            {"a": "A" * 100, "b": "B" * 200, "c": "C" * 300, "d": "D" * 400, "e": "E" * 500},
        )
        stats = compute_assembly_stats(str(fa))
        assert stats["n90"] == 200

    def test_l90_simple(self, tmp_path):
        fa = _write_fasta(
            tmp_path,
            {"a": "A" * 100, "b": "B" * 200, "c": "C" * 300, "d": "D" * 400, "e": "E" * 500},
        )
        stats = compute_assembly_stats(str(fa))
        # four contigs cover 1400/1500 ≥ 90%
        assert stats["l90"] == 4

    def test_single_contig_n50_equals_length(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "A" * 1000})
        stats = compute_assembly_stats(str(fa))
        assert stats["n50"] == 1000
        assert stats["l50"] == 1

    def test_n50_leq_n90(self, tmp_path):
        """N50 ≥ N90 by definition."""
        fa = _write_fasta(
            tmp_path,
            {str(i): "A" * (i * 50) for i in range(1, 11)},
        )
        stats = compute_assembly_stats(str(fa))
        assert stats["n50"] >= stats["n90"]


# ---------------------------------------------------------------------------
# format_assembly_stats
# ---------------------------------------------------------------------------


class TestFormatAssemblyStats:
    """format_assembly_stats returns a string containing expected fields."""

    def _base_stats(self):
        return {
            "n_contigs": 5,
            "total_length": 1500,
            "largest_contig": 500,
            "shortest_contig": 100,
            "mean_length": 300.0,
            "median_length": 300.0,
            "n50": 400,
            "l50": 2,
            "n90": 200,
            "l90": 4,
            "gc_content": 0.5,
        }

    def test_contains_n50(self):
        out = format_assembly_stats(self._base_stats())
        assert "N50" in out

    def test_contains_l50(self):
        out = format_assembly_stats(self._base_stats())
        assert "L50" in out

    def test_contains_n90(self):
        out = format_assembly_stats(self._base_stats())
        assert "N90" in out

    def test_contains_total_length(self):
        out = format_assembly_stats(self._base_stats())
        assert "1,500" in out

    def test_contains_gc(self):
        out = format_assembly_stats(self._base_stats())
        assert "%" in out

    def test_custom_label(self):
        out = format_assembly_stats(self._base_stats(), label="My Assembly")
        assert "My Assembly" in out

    def test_default_label(self):
        out = format_assembly_stats(self._base_stats())
        assert "Assembly statistics" in out

    def test_returns_string(self):
        out = format_assembly_stats(self._base_stats())
        assert isinstance(out, str)


# ---------------------------------------------------------------------------
# log_assembly_stats – smoke tests (does not crash)
# ---------------------------------------------------------------------------


class TestLogAssemblyStats:
    def test_smoke_plain_fasta(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "ACGT" * 50})
        log_assembly_stats(str(fa), label="Test run")  # must not raise

    def test_smoke_gzip_fasta(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "ACGT" * 50}, gz=True)
        log_assembly_stats(str(fa))  # must not raise

    def test_log_message_content(self, tmp_path, caplog):
        import logging

        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "G" * 200})
        with caplog.at_level(logging.INFO, logger="instagraal.assembly_stats"):
            log_assembly_stats(str(fa), label="My label")
        assert "N50" in caplog.text
        assert "My label" in caplog.text


# ---------------------------------------------------------------------------
# print_assembly_stats – always writes to stdout
# ---------------------------------------------------------------------------


class TestPrintAssemblyStats:
    def test_smoke(self, tmp_path):
        fa = _write_fasta(tmp_path, {"a": "ACGT" * 50})
        print_assembly_stats(str(fa), label="Test run")  # must not raise

    def test_output_goes_to_stdout(self, tmp_path, capsys):
        fa = _write_fasta(tmp_path, {"a": "A" * 100, "b": "G" * 200})
        print_assembly_stats(str(fa), label="My label")
        captured = capsys.readouterr()
        assert "N50" in captured.out
        assert "My label" in captured.out
        assert captured.err == ""


class TestFormatComparisonTable:
    def _stats_a(self):
        return {
            "n_contigs": 10,
            "total_length": 1_000_000,
            "largest_contig": 200_000,
            "shortest_contig": 5_000,
            "mean_length": 100_000.0,
            "median_length": 90_000.0,
            "n50": 150_000,
            "l50": 4,
            "n90": 60_000,
            "l90": 9,
            "gc_content": 0.45,
        }

    def _stats_b(self):
        return {
            "n_contigs": 5,
            "total_length": 800_000,
            "largest_contig": 300_000,
            "shortest_contig": 10_000,
            "mean_length": 160_000.0,
            "median_length": 140_000.0,
            "n50": 250_000,
            "l50": 2,
            "n90": 100_000,
            "l90": 4,
            "gc_content": 0.52,
        }

    def test_returns_string(self):
        out = format_comparison_table({"A": self._stats_a(), "B": self._stats_b()})
        assert isinstance(out, str)

    def test_contains_both_labels(self):
        out = format_comparison_table({"Assembly_A": self._stats_a(), "Assembly_B": self._stats_b()})
        assert "Assembly_A" in out
        assert "Assembly_B" in out

    def test_contains_n50_row(self):
        out = format_comparison_table({"A": self._stats_a(), "B": self._stats_b()})
        assert "N50" in out

    def test_contains_gc_row(self):
        out = format_comparison_table({"A": self._stats_a(), "B": self._stats_b()})
        assert "GC" in out or "gc" in out.lower()

    def test_single_entry_still_works(self):
        out = format_comparison_table({"only": self._stats_a()})
        assert "only" in out
        assert "N50" in out

    def test_three_columns(self):
        out = format_comparison_table(
            {
                "X": self._stats_a(),
                "Y": self._stats_b(),
                "Z": self._stats_a(),
            }
        )
        assert "X" in out
        assert "Y" in out
        assert "Z" in out


# ---------------------------------------------------------------------------
# instagraal-stats CLI
# ---------------------------------------------------------------------------


class TestStatsCLI:
    @pytest.fixture
    def runner(self):
        from click.testing import CliRunner

        return CliRunner()

    @pytest.fixture
    def fa_a(self, tmp_path):
        return _write_fasta(tmp_path, {"c1": "A" * 500, "c2": "G" * 300, "c3": "C" * 200})

    @pytest.fixture
    def fa_b(self, tmp_path):
        p = tmp_path / "b"
        p.mkdir()
        return _write_fasta(p, {"c1": "T" * 800, "c2": "A" * 400})

    def test_single_file_exit_code(self, runner, fa_a):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a)])
        assert result.exit_code == 0, result.output

    def test_single_file_output_has_n50(self, runner, fa_a):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a)])
        assert "N50" in result.output

    def test_single_file_shows_basename_as_label(self, runner, fa_a):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a)])
        assert fa_a.name in result.output

    def test_two_files_exit_code(self, runner, fa_a, fa_b):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a), str(fa_b)])
        assert result.exit_code == 0, result.output

    def test_two_files_output_has_both_basenames(self, runner, fa_a, fa_b):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a), str(fa_b)])
        assert fa_a.name in result.output
        assert fa_b.name in result.output

    def test_two_files_comparison_table(self, runner, fa_a, fa_b):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a), str(fa_b)])
        assert "comparison" in result.output.lower()
        assert "N50" in result.output

    def test_custom_labels(self, runner, fa_a, fa_b):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a), str(fa_b), "-l", "before,after"])
        assert result.exit_code == 0, result.output
        assert "before" in result.output
        assert "after" in result.output

    def test_label_count_mismatch_errors(self, runner, fa_a, fa_b):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [str(fa_a), str(fa_b), "-l", "only_one"])
        assert result.exit_code != 0

    def test_no_files_errors(self, runner):
        from instagraal.cli.stats import main

        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_gzip_input(self, runner, tmp_path):
        from instagraal.cli.stats import main

        fa_gz = _write_fasta(tmp_path, {"s1": "ACGT" * 100}, gz=True)
        result = runner.invoke(main, [str(fa_gz)])
        assert result.exit_code == 0, result.output
        assert "N50" in result.output
