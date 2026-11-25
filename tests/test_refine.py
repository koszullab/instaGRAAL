"""Tests for instagraal-refine (instagraal.refine and its CLI).

A synthetic circular chromosome is cut into contigs (two of them
reverse-complemented in the assembly) and Hi-C contacts are simulated with three
components: decay with distance along the circle, a strong stripe between mirror
loci of the two replication arms (the X that makes instaGRAAL fold chromosomes),
and background. The input layout handed to refine is folded on purpose.
A second, linear chromosome has a contig too short to be joined, which refine must
place from its contacts, wherever the input layout put it.
"""

import cooler
import numpy as np
import pandas as pd
import pytest
from Bio import SeqIO
from click.testing import CliRunner

from instagraal.cli.refine import main as refine_main
from instagraal.parse_info_frags import parse_info_frags, write_info_frags
from instagraal.refine import EndScores, build_scaffolds, gc_skew_summary, place_short_contigs, run_refine

GENOME_KB = 300
FRAG_KB = 1
CUTS_KB = [0, 60, 110, 170, 220, 260, 300]
REVERSED = {1, 4}  # contigs given reverse-complemented in the assembly
RC = str.maketrans("ACGT", "TGCA")


def _revcomp(seq):
    return seq.translate(RC)[::-1]


def _make_genome(d, cuts_kb, reversed_contigs, circular=True, mirror=2.0, seed=1):
    """Random chromosome cut into contigs (some reverse-complemented), with simulated contacts.

    Contacts between 1 kb fragments decay with distance (along the circle if
    ``circular``), plus a stripe of strength ``mirror`` between mirror loci of the two
    replication arms, plus background.
    """
    rng = np.random.default_rng(seed)
    genome_kb = cuts_kb[-1]
    chrom = "".join(rng.choice(list("ACGT"), size=genome_kb * 1000))
    names = [f"ctg{k}" for k in range(len(cuts_kb) - 1)]
    seqs = {}
    for k, name in enumerate(names):
        s = chrom[cuts_kb[k] * 1000 : cuts_kb[k + 1] * 1000]
        seqs[name] = _revcomp(s) if k in reversed_contigs else s
    fasta = d / "assembly.fa"
    fasta.write_text("".join(f">{n}\n{seqs[n]}\n" for n in names))

    # fragments of FRAG_KB, placed in contig coordinates (mirrored for reversed contigs)
    n_frag = genome_kb // FRAG_KB
    frag_contig, frag_start = [], []
    for f in range(n_frag):
        x = f * FRAG_KB
        k = max(i for i in range(len(names)) if cuts_kb[i] <= x)
        rel = x - cuts_kb[k]
        length = cuts_kb[k + 1] - cuts_kb[k]
        frag_contig.append(k)
        frag_start.append((length - rel - FRAG_KB) if k in reversed_contigs else rel)
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
    if circular:
        dist = np.minimum(dist, genome_kb - dist)
    mirror_dist = np.abs(pos[:, None] - (genome_kb - pos[None, :]) % genome_kb)
    mirror_dist = np.minimum(mirror_dist, genome_kb - mirror_dist)
    lam = 40.0 / np.maximum(dist, 1.0) + mirror * np.exp(-mirror_dist / 15.0) + 0.05
    counts = rng.poisson(np.triu(lam, 1))
    i, j = np.nonzero(counts)
    b1, b2 = bin_of[i], bin_of[j]
    pixels = pd.DataFrame({"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2), "count": counts[i, j]})
    pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    cool = d / "contacts.cool"
    cooler.create_cooler(str(cool), bins, pixels, ordered=True, symmetric_upper=True, dtypes={"count": np.int32})
    lengths = {n: (cuts_kb[k + 1] - cuts_kb[k]) * 1000 for k, n in enumerate(names)}
    return {"dir": d, "fasta": fasta, "cool": cool, "chrom": chrom, "lengths": lengths}


def _write_layout(path, scaffolds, lengths):
    """info_frags with whole contigs: ``scaffolds`` maps names to [(contig, orientation), ...]."""
    layout = {name: [[c, 0, 0, lengths[c], o] for c, o in entries] for name, entries in scaffolds.items()}
    write_info_frags(layout, output=str(path))
    return path


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    data = _make_genome(tmp_path_factory.mktemp("refine_synthetic"), CUTS_KB, REVERSED)
    # folded layout: the two arms interleaved, as instaGRAAL produces
    folded = {"scaffold1": [("ctg0", -1), ("ctg5", 1), ("ctg1", 1), ("ctg4", -1), ("ctg2", -1), ("ctg3", 1)]}
    data["layout"] = _write_layout(data["dir"] / "new_info_frags.txt", folded, data["lengths"])
    return data


# a linear chromosome with a 5 kb contig (ctg2, given reverse-complemented) between two long ones;
# 1 kb bins give ctg2 five bins, enough to orient it
SHORT_CUTS_KB = [0, 60, 110, 115, 170, 220, 260, 300]
SHORT_REVERSED = {1, 2, 5}
TRUE_ORDER = [("ctg0", 1), ("ctg1", -1), ("ctg2", -1), ("ctg3", 1), ("ctg4", 1), ("ctg5", -1), ("ctg6", 1)]


@pytest.fixture(scope="module")
def with_short_contig(tmp_path_factory):
    return _make_genome(tmp_path_factory.mktemp("refine_short"), SHORT_CUTS_KB, SHORT_REVERSED, circular=False, mirror=0.0)


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


# folded on purpose: ctg2 sits between ctg1 and ctg4, a wrong join; ctg3 is alone
WRONG_ORDER = {
    "scaffold1": [("ctg0", 1), ("ctg1", -1), ("ctg2", -1), ("ctg4", 1), ("ctg5", -1), ("ctg6", 1)],
    "scaffold2": [("ctg3", 1)],
}


def _refine_short(data, tmp_path, layout, cool=None, **kwargs):
    path = _write_layout(tmp_path / "layout.txt", layout, data["lengths"])
    kwargs = {"window": 30_000, "min_contig_length": 10_000, "binsize": 1_000, **kwargs}
    return run_refine(path, cool or data["cool"], data["fasta"], tmp_path / "out", **kwargs)


@pytest.mark.parametrize("layout", [{"scaffold1": TRUE_ORDER}, WRONG_ORDER], ids=["true-order", "wrong-order"])
def test_refine_places_short_contigs_from_their_contacts(with_short_contig, tmp_path, layout):
    """ctg2 (5 kb, too short to be joined) goes back between ctg1 and ctg3, in the right orientation."""
    data = with_short_contig
    summary = _refine_short(data, tmp_path, layout)
    assert summary["refined_joins"] == 5
    assert summary["short_placed"] == 1
    records = list(SeqIO.parse(str(tmp_path / "out" / "refined_genome.fa"), "fasta"))
    assert len(records) == 1
    seq = str(records[0].seq)
    assert seq == data["chrom"] or _revcomp(seq) == data["chrom"]
    table = pd.read_csv(tmp_path / "out" / "junctions.tsv", sep="\t")
    refined = table[table["layout"] == "refined"]
    assert (refined["status"] == "joined").sum() == 5
    placed = refined[refined["status"] == "placed"]
    assert len(placed) == 2
    assert (placed["end_a"].str.startswith("ctg2:") | placed["end_b"].str.startswith("ctg2:")).all()


def test_refine_leaves_short_contigs_without_contacts_alone(with_short_contig, tmp_path):
    data = with_short_contig
    clr = cooler.Cooler(str(data["cool"]))
    bins = clr.bins()[["chrom", "start", "end"]][:]
    pixels = clr.pixels()[:]
    ctg2 = set(np.nonzero((bins["chrom"] == "ctg2").to_numpy())[0])
    pixels = pixels[~pixels["bin1_id"].isin(ctg2) & ~pixels["bin2_id"].isin(ctg2)]
    cool = tmp_path / "no_ctg2_contacts.cool"
    cooler.create_cooler(str(cool), bins, pixels, ordered=True, symmetric_upper=True, dtypes={"count": np.int32})
    summary = _refine_short(data, tmp_path, {"scaffold1": TRUE_ORDER}, cool=cool)
    assert summary["short_placed"] == 0
    scaffolds = parse_info_frags(str(tmp_path / "out" / "refined_info_frags.txt"))
    assert [["ctg2"]] == [[b[0] for b in bins] for bins in scaffolds.values() if any(b[0] == "ctg2" for b in bins)]


def test_refine_no_place_short(with_short_contig, tmp_path):
    summary = _refine_short(with_short_contig, tmp_path, {"scaffold1": TRUE_ORDER}, place_short=False)
    assert summary["short_placed"] == 0
    assert summary["scaffolds"] == 2
    assert "each a scaffold of its own" in summary["report"]


def _scores(pairs):
    """EndScores over long contigs A, B and short ones s1, s2, s3, with the given symmetric scores."""
    ends = [(c, side) for c in ("A", "B", "s1", "s2", "s3") for side in ("L", "R")]
    index = {e: k for k, e in enumerate(ends)}
    score = np.zeros((len(ends), len(ends)))
    for (e1, e2), v in pairs.items():
        score[index[e1], index[e2]] = score[index[e2], index[e1]] = v
    return EndScores(ends, index, score, np.zeros_like(score), np.zeros_like(score), -1.0, 1.0)


def test_place_short_contigs_orders_and_orients_within_a_slot():
    es = _scores(
        {
            (("A", "R"), ("s1", "L")): 50.0,  # s1 right after A, forward
            (("s1", "R"), ("B", "L")): 5.0,
            (("s2", "L"), ("B", "L")): 40.0,  # s2 right before B, reversed
            (("A", "R"), ("s2", "R")): 5.0,
            (("B", "R"), ("s3", "L")): 30.0,  # s3 fits after B as well as before A: ambiguous
            (("s3", "R"), ("A", "L")): 30.0,
        }
    )
    chains, placed = place_short_contigs([[("A", 1), ("B", 1)]], es, ["s1", "s2", "s3"], {"A", "B"}, 10.0, 10.0)
    assert chains == [[("A", 1), ("s1", 1), ("s2", -1), ("B", 1)]]
    assert placed == {"s1": 55.0, "s2": 45.0}
