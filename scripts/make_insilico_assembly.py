#!/usr/bin/env python3
"""Generate an in silico contig-level assembly from a chromosome-level FASTA.

Simulates a realistic long-read assembly (ONT / PacBio HiFi) by fragmenting
chromosomes with a Poisson break process. A fixed random seed ensures
reproducibility.

Usage
-----
    python scripts/make_insilico_assembly.py example/yeast/S288c.fa yeast.contigs.fa.gz

Output
------
    <output_fasta>   FASTA with renamed contigs (contig_001, contig_002 …).
                     Each description records the source chromosome and the
                     coordinates of the slice: ``from_<chrom>:<s>-<e>``.
                     Output path ending in ``.gz`` is written gzip-compressed.
"""

import argparse
import gzip
import pathlib

import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ---------------------------------------------------------------------------
# Default parameters – tune these to change assembly characteristics
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
BREAK_RATE = 1.5    # expected breaks per Mb  →  realistic long-read assembly
                    # S. cerevisiae ~12.5 Mb × 1.5 ≈ 19 breaks → ~30 contigs
MIN_CONTIG_LEN = 1_000   # discard fragments shorter than this (bp)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def fragment_genome(
    fasta_path: str,
    break_rate: float = BREAK_RATE,
    min_len: int = MIN_CONTIG_LEN,
    seed: int = RANDOM_SEED,
) -> list[SeqRecord]:
    """Return a list of SeqRecords representing the fragmented assembly."""
    rng = np.random.default_rng(seed)
    contigs: list[SeqRecord] = []
    idx = 1

    opener = gzip.open if str(fasta_path).endswith(".gz") else open
    with opener(fasta_path, "rt") as fh:
        for rec in SeqIO.parse(fh, "fasta"):
            seq = str(rec.seq)
            chrom_len = len(seq)
            expected_breaks = break_rate * chrom_len / 1_000_000
            n_breaks = int(rng.poisson(expected_breaks))

            if n_breaks == 0:
                boundaries = [(0, chrom_len)]
            else:
                positions = sorted(rng.integers(1, chrom_len, n_breaks).tolist())
                boundaries = list(zip([0] + positions, positions + [chrom_len]))

            for start, end in boundaries:
                if end - start < min_len:
                    continue
                contigs.append(
                    SeqRecord(
                        Seq(seq[start:end]),
                        id=f"contig_{idx:03d}",
                        description=f"from_{rec.id}:{start}-{end}",
                    )
                )
                idx += 1

    return contigs


def print_stats(contigs: list[SeqRecord]) -> None:
    """Print N50/N90 and basic assembly metrics."""
    lengths = sorted([len(c.seq) for c in contigs], reverse=True)
    total = sum(lengths)
    cumsum = 0
    n50 = n90 = l50 = l90 = None

    for i, ln in enumerate(lengths, 1):
        cumsum += ln
        if n50 is None and cumsum >= total * 0.5:
            n50, l50 = ln, i
        if n90 is None and cumsum >= total * 0.9:
            n90, l90 = ln, i
            break

    print(f"  Sequences  : {len(contigs)}")
    print(f"  Total (bp) : {total:,}")
    print(f"  Largest    : {lengths[0]:,}")
    print(f"  Shortest   : {lengths[-1]:,}")
    print(f"  N50        : {n50:,}  (L50={l50})")
    print(f"  N90        : {n90:,}  (L90={l90})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_fasta",
        help="Input chromosome-level FASTA, plain or .gz (e.g. S288c.fa)",
    )
    parser.add_argument(
        "output_fasta",
        help="Output contig-level FASTA (use .gz suffix for gzip output)",
    )
    parser.add_argument(
        "--break-rate",
        type=float,
        default=BREAK_RATE,
        help=f"Expected breaks per Mb (default: {BREAK_RATE})",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=MIN_CONTIG_LEN,
        help=f"Minimum contig length to keep in bp (default: {MIN_CONTIG_LEN})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})",
    )
    args = parser.parse_args()

    print(f"Reading {args.input_fasta} …")
    contigs = fragment_genome(
        args.input_fasta,
        break_rate=args.break_rate,
        min_len=args.min_len,
        seed=args.seed,
    )

    out = pathlib.Path(args.output_fasta)
    opener = gzip.open if str(out).endswith(".gz") else open
    with opener(out, "wt") as fh:
        SeqIO.write(contigs, fh, "fasta")

    print(f"Written {len(contigs)} contigs → {out}")
    print_stats(contigs)


if __name__ == "__main__":
    main()
