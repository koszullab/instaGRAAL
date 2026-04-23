"""Utilities for computing and displaying assembly quality statistics.

Metrics are computed entirely from the FASTA sequences themselves - no
external databases or tools are required.

Glossary
--------
Nx  Length of the shortest contig in the sorted set whose cumulative
    length reaches x % of the total assembly length.
Lx  Number of contigs required to reach the Nx threshold.
"""

import gzip
import logging
import statistics

from Bio import SeqIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_lengths_and_gc(fasta_path: str) -> tuple[list[int], float]:
    """Return a list of sequence lengths and the overall GC fraction."""
    fasta_path = str(fasta_path)
    lengths: list[int] = []
    total_gc = 0
    total_bases = 0

    def _iter(handle, fmt="fasta"):
        yield from SeqIO.parse(handle, fmt)

    if fasta_path.endswith(".gz"):
        with gzip.open(fasta_path, "rt") as fh:
            records = list(_iter(fh))
    else:
        with open(fasta_path) as fh:
            records = list(_iter(fh))

    for record in records:
        seq = str(record.seq).upper()
        n = len(seq)
        if n == 0:
            continue
        lengths.append(n)
        gc = seq.count("G") + seq.count("C")
        total_gc += gc
        total_bases += n

    gc_fraction = total_gc / total_bases if total_bases > 0 else 0.0
    return lengths, gc_fraction


def _nx_lx(lengths_sorted_desc: list[int], total: int, fraction: float) -> tuple[int, int]:
    """Return (Nx, Lx) for a given *fraction* (0-1).

    *lengths_sorted_desc* must be sorted in descending order.
    """
    target = total * fraction
    cumulative = 0
    for idx, length in enumerate(lengths_sorted_desc, start=1):
        cumulative += length
        if cumulative >= target:
            return length, idx
    # Fallback - should not happen with a non-empty assembly
    return lengths_sorted_desc[-1], len(lengths_sorted_desc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_assembly_stats(fasta_path: str) -> dict:
    """Compute standard assembly statistics from a FASTA file.

    Parameters
    ----------
    fasta_path:
        Path to a plain or gzip-compressed FASTA file.

    Returns
    -------
    dict with keys:
        n_contigs, total_length, largest_contig, shortest_contig,
        mean_length, median_length, n50, l50, n90, l90, gc_content.
    """
    lengths, gc = _parse_lengths_and_gc(fasta_path)
    if not lengths:
        return {
            "n_contigs": 0,
            "total_length": 0,
            "largest_contig": 0,
            "shortest_contig": 0,
            "mean_length": 0.0,
            "median_length": 0.0,
            "n50": 0,
            "l50": 0,
            "n90": 0,
            "l90": 0,
            "gc_content": 0.0,
        }

    lengths_desc = sorted(lengths, reverse=True)
    total = sum(lengths)
    n50, l50 = _nx_lx(lengths_desc, total, 0.50)
    n90, l90 = _nx_lx(lengths_desc, total, 0.90)

    return {
        "n_contigs": len(lengths),
        "total_length": total,
        "largest_contig": lengths_desc[0],
        "shortest_contig": lengths_desc[-1],
        "mean_length": total / len(lengths),
        "median_length": statistics.median(lengths),
        "n50": n50,
        "l50": l50,
        "n90": n90,
        "l90": l90,
        "gc_content": gc,
    }


def format_assembly_stats(stats: dict, label: str | None = None) -> str:
    """Return a human-readable, multi-line summary of assembly statistics.

    Parameters
    ----------
    stats:
        Dictionary as returned by :func:`compute_assembly_stats`.
    label:
        Optional heading printed above the table.
    """
    header = f"  {label}" if label else "  Assembly statistics"
    sep = "  " + "-" * 42
    lines = [
        "",
        header,
        sep,
        f"  {'Sequences (contigs/scaffolds)':<34} {stats['n_contigs']:>7,}",
        f"  {'Total length (bp)':<34} {stats['total_length']:>7,}",
        f"  {'Largest sequence (bp)':<34} {stats['largest_contig']:>7,}",
        f"  {'Shortest sequence (bp)':<34} {stats['shortest_contig']:>7,}",
        f"  {'Mean length (bp)':<34} {stats['mean_length']:>10,.1f}",
        f"  {'Median length (bp)':<34} {stats['median_length']:>10,.1f}",
        f"  {'N50 (bp)':<34} {stats['n50']:>7,}",
        f"  {'L50 (# sequences)':<34} {stats['l50']:>7,}",
        f"  {'N90 (bp)':<34} {stats['n90']:>7,}",
        f"  {'L90 (# sequences)':<34} {stats['l90']:>7,}",
        f"  {'GC content':<34} {stats['gc_content'] * 100:>9.2f}%",
        sep,
        "",
    ]
    return "\n".join(lines)


# Row definitions for the comparison table (label, key, formatter)
_COMPARISON_ROWS: list[tuple[str, str, str]] = [
    ("Sequences", "n_contigs", "int"),
    ("Total length (bp)", "total_length", "int"),
    ("Largest (bp)", "largest_contig", "int"),
    ("Shortest (bp)", "shortest_contig", "int"),
    ("Mean length (bp)", "mean_length", "float"),
    ("Median length (bp)", "median_length", "float"),
    ("N50 (bp)", "n50", "int"),
    ("L50", "l50", "int"),
    ("N90 (bp)", "n90", "int"),
    ("L90", "l90", "int"),
    ("GC content", "gc_content", "pct"),
]


def format_comparison_table(results: dict) -> str:
    """Return a side-by-side comparison table for multiple assemblies.

    Parameters
    ----------
    results:
        Ordered mapping of ``{label: stats_dict}`` where each ``stats_dict``
        is as returned by :func:`compute_assembly_stats`.  Pass an
        ``collections.OrderedDict`` or a plain ``dict`` (Python 3.7+ preserves
        insertion order) to control column order.

    Returns
    -------
    A multi-line string ready to be printed or logged.
    """
    labels = list(results.keys())
    col_w = max(14, *(len(lb) + 2 for lb in labels))
    metric_w = 22

    def _fmt(value, kind: str) -> str:
        if kind == "int":
            return f"{int(value):>{col_w},}"
        if kind == "float":
            return f"{value:>{col_w},.1f}"
        if kind == "pct":
            return f"{value * 100:>{col_w - 1}.2f}%"
        return str(value)

    sep = "-" * (metric_w + col_w * len(labels) + 2)
    header_row = f"{'Metric':<{metric_w}}" + "".join(f"{lb:>{col_w}}" for lb in labels)

    rows = [
        "",
        "  Assembly statistics comparison",
        "  " + sep,
        "  " + header_row,
        "  " + sep,
    ]
    for row_label, key, kind in _COMPARISON_ROWS:
        values = "".join(_fmt(results[lb][key], kind) for lb in labels)
        rows.append(f"  {row_label:<{metric_w}}{values}")
    rows += ["  " + sep, ""]
    return "\n".join(rows)


def print_assembly_stats(fasta_path: str, label: str | None = None) -> None:
    """Compute and print assembly statistics for *fasta_path* to stdout.

    Always prints regardless of logging configuration – intended for use in
    CLI commands that should unconditionally display stats to the user.

    Parameters
    ----------
    fasta_path:
        Path to a plain or gzip-compressed FASTA file.
    label:
        Optional heading for the statistics block.
    """
    try:
        stats = compute_assembly_stats(fasta_path)
        print(format_assembly_stats(stats, label=label))
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not compute assembly stats for {fasta_path}: {exc}")


def log_assembly_stats(fasta_path: str, label: str | None = None) -> None:
    """Compute and log assembly statistics for *fasta_path* at INFO level.

    Parameters
    ----------
    fasta_path:
        Path to a plain or gzip-compressed FASTA file.
    label:
        Optional heading for the statistics block.
    """
    try:
        stats = compute_assembly_stats(fasta_path)
        msg = format_assembly_stats(stats, label=label)
        logger.info(msg)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not compute assembly stats for %s: %s", fasta_path, exc)
