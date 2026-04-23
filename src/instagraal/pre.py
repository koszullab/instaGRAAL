#!/usr/bin/env python3
"""Pre-processing module for instaGRAAL.

Converts a FASTA genome and a Hi-C pairs file (4DN pairs format, columns:
readID chr1 pos1 chr2 pos2 strand1 strand2) into the three text files
required by instaGRAAL:

  - fragments_list.txt
  - info_contigs.txt
  - abs_fragments_contacts_weighted.txt

A .cool file is also written as a useful intermediate / final output.

Usage example
-------------
    instagraal-pre genome.fa contacts.pairs \\
        --enzyme DpnII,HinfI \\
        --output-dir ./hic_folder
"""

import gzip
import pathlib

import Bio.Restriction as biorst
import Bio.Seq as bioseq
import cooler
import numpy as np
import pandas as pd
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def _parse_fasta(fasta_path: pathlib.Path) -> dict[str, str]:
    """Load a FASTA file into a dict mapping contig name → sequence string.

    Handles both plain (``.fa``, ``.fasta``) and gzip-compressed
    (``.fa.gz``, ``.fasta.gz``) files transparently.
    """
    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    with opener(fasta_path, "rt") as fh:
        return {rec.id: str(rec.seq) for rec in SeqIO.parse(fh, "fasta")}


def _multi_enzyme_digest(fasta_records: dict[str, str], enzymes: list[str]) -> pd.DataFrame:
    """Digest a genome with one or more restriction enzymes.

    Cut sites from all enzymes are merged, deduplicated and sorted before
    fragments are built.  This replicates what ``cooler digest`` does for a
    single enzyme but extends it to an arbitrary number of enzymes.

    Parameters
    ----------
    fasta_records:
        Dict mapping contig name to sequence string.
    enzymes:
        List of restriction enzyme names recognised by Biopython (e.g.
        ``["DpnII", "HinfI"]``).

    Returns
    -------
    pd.DataFrame
        Columns: ``chrom``, ``start``, ``end`` (0-based, half-open).
    """
    # Validate enzymes up front
    cut_finders = []
    for name in enzymes:
        try:
            cut_finders.append(getattr(biorst, name).search)
        except AttributeError:
            raise ValueError(f"Unknown restriction enzyme: {name!r}")

    frames = []
    for chrom, chrom_seq in fasta_records.items():
        seq = bioseq.Seq(chrom_seq)
        # Collect cuts from every enzyme (1-based positions from biopython)
        all_cuts = set()
        for finder in cut_finders:
            all_cuts.update(finder(seq))
        # biopython search() returns 1-based position of the first base of the
        # recognition site (i.e. the first base of the next fragment). Convert
        # to 0-based by subtracting 1. Then add sentinels 0 and len(seq).
        cuts = np.array(sorted(all_cuts), dtype=np.int64) - 1
        cuts = np.unique(np.r_[0, cuts, len(seq)].astype(np.int64))
        n_frags = len(cuts) - 1
        frames.append(
            pd.DataFrame(
                {
                    "chrom": [chrom] * n_frags,
                    "start": cuts[:-1],
                    "end": cuts[1:],
                }
            )
        )

    return pd.concat(frames, axis=0, ignore_index=True)


def _gc_content(seq: str) -> float:
    """Fraction of G+C bases in *seq*. Returns 0.0 for empty sequences."""
    if not seq:
        return 0.0
    seq_upper = seq.upper()
    gc = seq_upper.count("G") + seq_upper.count("C")
    return gc / len(seq_upper)


def _build_bins_with_gc(bins: pd.DataFrame, fasta_records: dict[str, str]) -> pd.DataFrame:
    """Attach a ``gc_content`` column to a bins DataFrame.

    Parameters
    ----------
    bins:
        DataFrame with ``chrom``, ``start``, ``end`` columns.
    fasta_records:
        Dict mapping contig name to sequence string.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with an additional ``gc_content`` column.
    """
    gc_values = []
    for row in bins.itertuples(index=False):
        seq = fasta_records[row.chrom][row.start : row.end]
        gc_values.append(_gc_content(seq))
    bins = bins.copy()
    bins["gc_content"] = gc_values
    return bins


def _pairs_to_pixels(pairs_path: pathlib.Path, bins: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Read a pairs file and bin reads into fragment pixel counts.

    Handles both plain text and gzip-compressed pairs files.  The function
    reads the ``#columns:`` header line to locate chr1/pos1/chr2/pos2 fields
    dynamically, so it is robust to different column orderings.

    Parameters
    ----------
    pairs_path:
        Path to the pairs file (plain or .gz).
    bins:
        DataFrame with ``chrom``, ``start``, ``end`` indexed from 0.

    Returns
    -------
    pixels : pd.DataFrame
        Columns: ``bin1_id``, ``bin2_id``, ``count`` (upper-triangular,
        0-indexed).
    total_contacts : int
        Total number of read pairs processed.
    """
    # Build a lookup: (chrom, pos) → bin_id using interval membership.
    # We use a per-chrom sorted array of starts for fast searchsorted lookup.
    chrom_to_starts = {}
    chrom_to_bin_offset = {}
    cumul = 0
    for chrom, grp in bins.groupby("chrom", sort=False):
        starts = grp["start"].values
        chrom_to_starts[chrom] = starts
        chrom_to_bin_offset[chrom] = (cumul, grp.index.values)
        cumul += len(starts)

    def _pos_to_bin_id(chrom: str, pos: int) -> int | None:
        """Return 0-based bin id for a 1-based genomic position."""
        if chrom not in chrom_to_starts:
            return None
        starts = chrom_to_starts[chrom]
        # Pairs positions are 1-based; convert to 0-based before searchsorted
        # so that a read at the last base of a fragment (1-based pos == cut
        # site value) is correctly assigned to the left fragment, matching
        # hicstuff's attribute_fragments logic (which does pos - 1 before the
        # binary search).
        idx = np.searchsorted(starts, pos - 1, side="right") - 1
        if idx < 0:
            return None
        _, bin_indices = chrom_to_bin_offset[chrom]
        return int(bin_indices[idx])

    # Detect columns from header
    col1_idx, col2_idx, col3_idx, col4_idx = 1, 2, 3, 4  # defaults
    opener = gzip.open if str(pairs_path).endswith((".gz", ".bgz")) else open

    counts: dict[tuple[int, int], int] = {}
    total = 0

    with opener(pairs_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("#columns:"):
                    cols = line.strip().split()[1:]  # drop "#columns:"
                    try:
                        col1_idx = cols.index("chr1")
                        col2_idx = cols.index("pos1")
                        col3_idx = cols.index("chr2")
                        col4_idx = cols.index("pos2")
                    except ValueError:
                        pass  # keep defaults
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                chr1 = parts[col1_idx]
                pos1 = int(parts[col2_idx])
                chr2 = parts[col3_idx]
                pos2 = int(parts[col4_idx])
            except (IndexError, ValueError):
                continue

            b1 = _pos_to_bin_id(chr1, pos1)
            b2 = _pos_to_bin_id(chr2, pos2)
            if b1 is None or b2 is None:
                continue

            total += 1
            key = (min(b1, b2), max(b1, b2))
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        pixels = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])
    else:
        keys = sorted(counts.keys())
        bin1_ids, bin2_ids = zip(*keys)
        pixel_counts = [counts[k] for k in keys]
        pixels = pd.DataFrame(
            {
                "bin1_id": np.array(bin1_ids, dtype=np.int32),
                "bin2_id": np.array(bin2_ids, dtype=np.int32),
                "count": np.array(pixel_counts, dtype=np.int32),
            }
        )

    return pixels, total


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _write_fragments_list(bins: pd.DataFrame, output_path: pathlib.Path) -> None:
    """Write ``fragments_list.txt`` from a bins DataFrame with gc_content.

    Format (tab-separated, per-contig 1-indexed IDs)::

        id  chrom  start_pos  end_pos  size  gc_content

    The ``id`` column resets to 1 for every new contig, matching the
    convention expected by instaGRAAL's pyramid-building code.
    """
    with open(output_path, "w") as fh:
        fh.write("id\tchrom\tstart_pos\tend_pos\tsize\tgc_content\n")
        for chrom, grp in bins.groupby("chrom", sort=False):
            for i, row in enumerate(grp.itertuples(index=False), start=1):
                size = row.end - row.start
                fh.write(f"{i}\t{row.chrom}\t{row.start}\t{row.end}\t{size}\t{row.gc_content}\n")


def _write_info_contigs(bins: pd.DataFrame, fasta_records: dict[str, str], output_path: pathlib.Path) -> None:
    """Write ``info_contigs.txt``.

    Format (tab-separated)::

        contig  length  n_frags  cumul_length
    """
    with open(output_path, "w") as fh:
        fh.write("contig\tlength\tn_frags\tcumul_length\n")
        cumul = 0
        for chrom, grp in bins.groupby("chrom", sort=False):
            length = len(fasta_records[chrom])
            n = len(grp)
            fh.write(f"{chrom}\t{length}\t{n}\t{cumul}\n")
            cumul += n


def _write_abs_contacts(pixels: pd.DataFrame, n_frags: int, output_path: pathlib.Path) -> None:
    """Write ``abs_fragments_contacts_weighted.txt``.

    Format (tab-separated)::

        <nfrags>  <nfrags>  <n_pixels>    ← header (n_pixels = nonzero entries)
        bin1_id   bin2_id   count         ← rows (0-indexed, upper-tri)
    """
    n_pixels = len(pixels)
    with open(output_path, "w") as fh:
        fh.write(f"{n_frags}\t{n_frags}\t{n_pixels}\n")
        for row in pixels.itertuples(index=False):
            fh.write(f"{row.bin1_id}\t{row.bin2_id}\t{row.count}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pre(
    fasta: pathlib.Path,
    pairs: pathlib.Path,
    enzymes: list[str],
    output_dir: pathlib.Path,
    cool_name: str | None = None,
) -> None:
    """Run the full pre-processing pipeline.

    Parameters
    ----------
    fasta:
        Path to the genome FASTA file.
    pairs:
        Path to the Hi-C pairs file (plain or gzip, 4DN format).
    enzymes:
        List of restriction enzyme names.
    output_dir:
        Directory where output files will be written.
    cool_name:
        Base name (without extension) for the output .cool file.
        Defaults to the pairs file stem.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Opening genome: {fasta}")
    fasta_records = _parse_fasta(fasta)

    print(f"[2/5] Digesting genome with enzyme(s): {', '.join(enzymes)}")
    bins = _multi_enzyme_digest(fasta_records, enzymes)
    n_frags = len(bins)
    print(f"      → {n_frags:,} restriction fragments")

    print("[3/5] Computing GC content per fragment")
    bins = _build_bins_with_gc(bins, fasta_records)

    print(f"[4/5] Binning pairs into fragment contact matrix: {pairs}")
    pixels, total_contacts = _pairs_to_pixels(pairs, bins)
    print(f"      → {total_contacts:,} valid pairs, {len(pixels):,} non-zero pixels")

    # Write .cool
    if cool_name is None:
        # Strip up to two extensions to handle .pairs and .pairs.gz
        stem = pairs.name
        for _ in range(2):
            stem, ext = pathlib.Path(stem).stem, pathlib.Path(stem).suffix
            if not ext:
                break
        cool_name = stem
    cool_path = output_dir / f"{cool_name}.cool"
    print(f"[5/5] Writing outputs to {output_dir}")
    print(f"      → {cool_path.name}")

    # Build chromsizes Series in contig order
    bins_for_cooler = bins[["chrom", "start", "end"]].copy()
    cooler.create_cooler(
        str(cool_path),
        bins=bins_for_cooler,
        pixels=pixels,
        dtypes={"count": np.int32},
        assembly=fasta.stem.removesuffix(".fa").removesuffix(".fasta"),
        ordered=True,
        symmetric_upper=True,
    )

    frags_path = output_dir / "fragments_list.txt"
    print(f"      → {frags_path.name}")
    _write_fragments_list(bins, frags_path)

    contigs_path = output_dir / "info_contigs.txt"
    print(f"      → {contigs_path.name}")
    _write_info_contigs(bins, fasta_records, contigs_path)

    contacts_path = output_dir / "abs_fragments_contacts_weighted.txt"
    print(f"      → {contacts_path.name}")
    _write_abs_contacts(pixels, n_frags, contacts_path)

    print("Done.")


if __name__ == "__main__":
    from .cli.pre import main

    main()
