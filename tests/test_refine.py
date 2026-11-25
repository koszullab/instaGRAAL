"""Tests for instagraal-refine (instagraal.refine and its CLI).

A synthetic circular chromosome is cut into contigs (two of them
reverse-complemented in the assembly) and Hi-C contacts are simulated with three
components: decay with distance along the circle, a strong stripe between mirror
loci of the two replication arms (the X that makes instaGRAAL fold chromosomes),
and background. The input layout handed to refine is folded on purpose.
"""

import cooler
import numpy as np
import pandas as pd
import pytest
from Bio import SeqIO
from click.testing import CliRunner

from instagraal.cli.refine import main as refine_main
from instagraal.parse_info_frags import parse_info_frags, write_info_frags
from instagraal.refine import build_scaffolds, gc_skew_summary, run_refine

GENOME_KB = 300
FRAG_KB = 1
CUTS_KB = [0, 60, 110, 170, 220, 260, 300]
REVERSED = {1, 4}  # contigs given reverse-complemented in the assembly
RC = str.maketrans("ACGT", "TGCA")


def _revcomp(seq):
    return seq.translate(RC)[::-1]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    rng = np.random.default_rng(1)
    d = tmp_path_factory.mktemp("refine_synthetic")
    chrom = "".join(rng.choice(list("ACGT"), size=GENOME_KB * 1000))
    names = [f"ctg{k}" for k in range(len(CUTS_KB) - 1)]
    seqs = {}
    for k, name in enumerate(names):
        s = chrom[CUTS_KB[k] * 1000 : CUTS_KB[k + 1] * 1000]
        seqs[name] = _revcomp(s) if k in REVERSED else s
    fasta = d / "assembly.fa"
    fasta.write_text("".join(f">{n}\n{seqs[n]}\n" for n in names))

    # fragments of FRAG_KB, placed in contig coordinates (mirrored for reversed contigs)
    n_frag = GENOME_KB // FRAG_KB
    frag_contig, frag_start = [], []
    for f in range(n_frag):
        x = f * FRAG_KB
        k = max(i for i in range(len(names)) if CUTS_KB[i] <= x)
        rel = x - CUTS_KB[k]
        length = CUTS_KB[k + 1] - CUTS_KB[k]
        frag_contig.append(k)
        frag_start.append((length - rel - FRAG_KB) if k in REVERSED else rel)
    order = sorted(range(n_frag), key=lambda f: (frag_contig[f], frag_start[f]))
    bin_of = np.empty(n_frag, dtype=int)
    bin_of[order] = np.arange(n_frag)
    bins = pd.DataFrame(
        {
            "chrom": [names[frag_contig[f]] for f in order],
            "start": [frag_start[f] * 1000 for f in order],
            "end": [(frag_start[f] + FRAG_KB) * 1000 for f in order],
        }
    )
    pos = (np.arange(n_frag) + 0.5) * FRAG_KB
    dist = np.abs(pos[:, None] - pos[None, :])
    dist = np.minimum(dist, GENOME_KB - dist)
    mirror = np.abs(pos[:, None] - (GENOME_KB - pos[None, :]) % GENOME_KB)
    mirror = np.minimum(mirror, GENOME_KB - mirror)
    lam = 40.0 / np.maximum(dist, 1.0) + 2.0 * np.exp(-mirror / 15.0) + 0.05
    counts = rng.poisson(np.triu(lam, 1))
    i, j = np.nonzero(counts)
    b1, b2 = bin_of[i], bin_of[j]
    pixels = pd.DataFrame({"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2), "count": counts[i, j]})
    pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    cool = d / "contacts.cool"
    cooler.create_cooler(str(cool), bins, pixels, ordered=True, symmetric_upper=True, dtypes={"count": np.int32})

    # folded layout: the two arms interleaved, as instaGRAAL produces
    folded = {
        "scaffold1": [
            ["ctg0", 0, 0, 60_000, -1],
            ["ctg5", 0, 0, 40_000, 1],
            ["ctg1", 0, 0, 50_000, 1],
            ["ctg4", 0, 0, 40_000, -1],
            ["ctg2", 0, 0, 60_000, -1],
            ["ctg3", 0, 0, 50_000, 1],
        ]
    }
    layout = d / "new_info_frags.txt"
    write_info_frags(folded, output=str(layout))
    return {"dir": d, "fasta": fasta, "cool": cool, "layout": layout, "chrom": chrom}


def _refine(synthetic, out, **kwargs):
    kwargs = {"window": 30_000, "min_contig_length": 10_000, **kwargs}
    return run_refine(synthetic["layout"], synthetic["cool"], synthetic["fasta"], out, **kwargs)


def test_refine_recovers_the_chromosome(synthetic, tmp_path):
    summary = _refine(synthetic, tmp_path)
    records = list(SeqIO.parse(str(tmp_path / "refined_genome.fa"), "fasta"))
    assert len(records) == 1, "all contigs should be joined into one scaffold"
    seq = str(records[0].seq)
    assert len(seq) == GENOME_KB * 1000
    circle = synthetic["chrom"] * 2
    assert seq in circle or _revcomp(seq) in circle, "refined scaffold is not a rotation of the chromosome"
    assert summary["refined_joins"] == len(CUTS_KB) - 2


def test_refine_flags_the_folded_layout(synthetic, tmp_path):
    summary = _refine(synthetic, tmp_path)
    statuses = summary["input_statuses"]
    assert statuses.get("kept", 0) == 0
    assert statuses.get("contradicted", 0) + statuses.get("unsupported", 0) == summary["input_joins"]
    table = pd.read_csv(tmp_path / "junctions.tsv", sep="\t")
    assert set(table["layout"]) == {"input", "refined"}
    assert (table.loc[table["layout"] == "refined", "status"] == "joined").all()
    assert (tmp_path / "refine_report.txt").read_text().startswith("instagraal-refine")


def test_refined_layout_is_usable_downstream(synthetic, tmp_path):
    """refined_info_frags.txt holds whole contigs, each once, and matches the FASTA (with junctions)."""
    _refine(synthetic, tmp_path, junction="NNNNNN")
    scaffolds = parse_info_frags(str(tmp_path / "refined_info_frags.txt"))
    used = [b[0] for bins in scaffolds.values() for b in bins]
    assert sorted(used) == sorted(f"ctg{k}" for k in range(len(CUTS_KB) - 1))
    fa = {r.id: len(r.seq) for r in SeqIO.parse(str(tmp_path / "refined_genome.fa"), "fasta")}
    for name, bins in scaffolds.items():
        assert fa[name] == sum(e - s for _c, _i, s, e, _o in bins) + 6 * (len(bins) - 1)


def test_refine_qc_only_writes_no_assembly(synthetic, tmp_path):
    summary = _refine(synthetic, tmp_path, qc_only=True)
    assert (tmp_path / "junctions.tsv").exists()
    assert (tmp_path / "refine_report.txt").exists()
    assert not (tmp_path / "refined_genome.fa").exists()
    assert summary["input_joins"] == len(CUTS_KB) - 2


def test_refine_cli(synthetic, tmp_path):
    out = tmp_path / "out"
    result = CliRunner().invoke(
        refine_main,
        [
            "-i",
            str(synthetic["layout"]),
            "-c",
            str(synthetic["cool"]),
            "-f",
            str(synthetic["fasta"]),
            "-o",
            str(out),
            "-w",
            "30000",
            "-l",
            "10000",
            "-j",
            "NNNNNN",
        ],
    )
    assert result.exit_code == 0, result.output
    assert f"Refined: {len(CUTS_KB) - 2} joins" in result.output
    assert (out / "refined_info_frags.txt").exists()


def test_refine_rejects_contigs_missing_from_the_contact_map(synthetic, tmp_path):
    fasta = tmp_path / "more.fa"
    fasta.write_text(synthetic["fasta"].read_text() + ">not_in_cool\nACGTACGT\n")
    result = CliRunner().invoke(
        refine_main, ["-i", str(synthetic["layout"]), "-c", str(synthetic["cool"]), "-f", str(fasta), "-o", str(tmp_path / "o")]
    )
    assert result.exit_code != 0
    assert "absent from the contact map" in result.output


def test_build_scaffolds_reads_chains_with_orientations():
    joins = [(("A", "R"), ("B", "R")), (("B", "L"), ("C", "L"))]
    chains = build_scaffolds(["B", "D", "A", "C"], joins)
    assert [("C", -1), ("B", 1), ("A", -1)] in chains or [("A", 1), ("B", -1), ("C", 1)] in chains
    assert [("D", 1)] in chains


def test_gc_skew_summary_on_two_replichores():
    rng = np.random.default_rng(0)
    # leading-strand bias on the first half (G > C), reversed on the second half
    first = "".join(rng.choice(list("GGGCAT"), size=300_000))
    second = first.translate(str.maketrans("GC", "CG"))
    share, changes = gc_skew_summary(first + second)
    assert share == pytest.approx(0.5, abs=0.05)
    assert changes == 1
    share_folded, changes_folded = gc_skew_summary(first + first)
    assert share_folded > 0.95 and changes_folded == 0
