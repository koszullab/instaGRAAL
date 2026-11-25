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
import instagraal.refine as refine_module
from instagraal.refine import (
    EndScores,
    build_scaffolds,
    gc_skew_summary,
    place_short_contigs,
    replication_axis,
    rotate,
    run_refine,
)

GENOME_KB = 300
FRAG_KB = 1
CUTS_KB = [0, 60, 110, 170, 220, 260, 300]
REVERSED = {1, 4}  # contigs given reverse-complemented in the assembly
RC = str.maketrans("ACGT", "TGCA")


def _revcomp(seq):
    return seq.translate(RC)[::-1]


def _make_genome(d, cuts_kb, reversed_contigs, circular=True, mirror=2.0, seed=1, skew=0.0):
    """Random chromosome cut into contigs (some reverse-complemented), with simulated contacts.

    Contacts between 1 kb fragments decay with distance (along the circle if
    ``circular``), plus a stripe of strength ``mirror`` between mirror loci of the two
    replication arms, plus background. With ``skew``, the first half of the chromosome
    has GC skew ``skew`` and the second half ``-skew``: origin at 0, terminus in the middle.
    """
    rng = np.random.default_rng(seed)
    genome_kb = cuts_kb[-1]
    if skew:
        half = genome_kb * 500
        g_rich = [0.25, 0.25 - skew / 4, 0.25 + skew / 4, 0.25]  # A, C, G, T
        chrom = "".join(rng.choice(list("ACGT"), size=half, p=g_rich)) + "".join(
            rng.choice(list("ACGT"), size=half, p=g_rich[:1] + g_rich[2:0:-1] + g_rich[3:])
        )
    else:
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
    assert summary["circular"] == []  # 300 kb: below CIRCULAR_MIN_LENGTH, not tested
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


# ---------------------------------------------------------------------------
# Circular scaffolds and recentering
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def skewed_circle(tmp_path_factory):
    """The folded synthetic circle, with GC skew: origin at 0, terminus at 150 kb."""
    data = _make_genome(tmp_path_factory.mktemp("refine_skewed"), CUTS_KB, REVERSED, skew=0.2)
    folded = {"scaffold1": [("ctg0", -1), ("ctg5", 1), ("ctg1", 1), ("ctg4", -1), ("ctg2", -1), ("ctg3", 1)]}
    data["layout"] = _write_layout(data["dir"] / "new_info_frags.txt", folded, data["lengths"])
    return data


def _start_on_circle(seq, chrom):
    """Position of the circle (forward strand) where a rotation of it, on either strand, starts."""
    size = len(chrom)
    forward = (chrom * 2).find(seq[:2000])
    if forward >= 0:
        return forward % size
    return (size - (_revcomp(chrom) * 2).find(seq[:2000])) % size


def _circular_distance(a, b, size):
    d = abs(a - b) % size
    return min(d, size - d)


@pytest.mark.parametrize("origin_first,expected_start", [(False, 150_000), (True, 0)], ids=["terminus-cut", "origin-first"])
def test_refine_recenters_a_circular_chromosome(skewed_circle, tmp_path, monkeypatch, origin_first, expected_start):
    monkeypatch.setattr(refine_module, "CIRCULAR_MIN_LENGTH", 100_000)
    data = skewed_circle
    summary = _refine(data, tmp_path, origin_first=origin_first)
    assert summary["circular"] == summary["recentered"] == ["refined_scaffold_1"]
    records = list(SeqIO.parse(str(tmp_path / "refined_genome.fa"), "fasta"))
    assert len(records) == 1
    seq = str(records[0].seq)
    assert len(seq) == GENOME_KB * 1000
    assert seq in data["chrom"] * 2 or _revcomp(seq) in data["chrom"] * 2
    # cut at the terminus (origin in the middle), or at the origin: within one 10 kb skew window
    assert _circular_distance(_start_on_circle(seq, data["chrom"]), expected_start, len(data["chrom"])) <= 10_000
    assert "recentered on its GC skew" in summary["report"]
    # the contig holding the cut starts and ends the scaffold, and the layout still matches the FASTA
    scaffolds = parse_info_frags(str(tmp_path / "refined_info_frags.txt"))
    (bins,) = scaffolds.values()
    assert sum(e - s for _c, _i, s, e, _o in bins) == len(seq)


def test_refine_leaves_skewless_circles_alone(synthetic, tmp_path, monkeypatch):
    """The plain synthetic circle has no GC skew: circular, but not recentered."""
    monkeypatch.setattr(refine_module, "CIRCULAR_MIN_LENGTH", 100_000)
    summary = _refine(synthetic, tmp_path)
    assert summary["circular"] == ["refined_scaffold_1"]
    assert summary["recentered"] == []
    assert "not recentered" in summary["report"]
    # the join refine refuses (it would close a cycle) is reported
    table = pd.read_csv(tmp_path / "junctions.tsv", sep="\t")
    assert list(table.loc[table["status"] == "closure", "scaffold"]) == ["refined_scaffold_1"]


def test_refine_no_recenter(skewed_circle, tmp_path, monkeypatch):
    monkeypatch.setattr(refine_module, "CIRCULAR_MIN_LENGTH", 100_000)
    summary = _refine(skewed_circle, tmp_path, recenter=False)
    assert summary["circular"] == ["refined_scaffold_1"]
    assert summary["recentered"] == []


def test_linear_chromosome_is_not_circular(with_short_contig, tmp_path):
    summary = _refine_short(with_short_contig, tmp_path, {"scaffold1": TRUE_ORDER})
    assert summary["circular"] == []


def test_replication_axis_on_two_replichores():
    rng = np.random.default_rng(0)
    g_rich = "".join(rng.choice(list("ACGT"), size=300_000, p=[0.25, 0.2, 0.3, 0.25]))
    c_rich = g_rich.translate(str.maketrans("GC", "CG"))
    chrom = g_rich + c_rich  # origin at 0, terminus at 300 kb
    size = len(chrom)
    for shift in (0, 150_000, 420_000):
        rotated = chrom[shift:] + chrom[:shift]
        for seq in (rotated, _revcomp(rotated)):
            origin, terminus, fit = replication_axis(seq)
            expected_origin = (-shift) % size if seq is rotated else shift % size
            assert _circular_distance(origin, expected_origin, size) <= 10_000
            assert _circular_distance(terminus, (expected_origin + 300_000) % size, size) <= 10_000
            assert fit > 0.9
    assert replication_axis("ACGT" * 1000) is None


def test_rotate_splits_the_contig_holding_the_cut():
    rng = np.random.default_rng(3)
    genome = {name: "".join(rng.choice(list("ACGT"), size=n)) for name, n in (("a", 30), ("b", 50), ("c", 20))}
    entries = [("a", 1, 0, 30), ("b", -1, 0, 50), ("c", 1, 5, 20)]

    def sequence(es):
        return "".join(genome[c][s:e] if o == 1 else _revcomp(genome[c][s:e]) for c, o, s, e in es)

    whole = sequence(entries)
    for cut in (0, 10, 30, 47, 80, 94):
        rotated = rotate(entries, cut)
        assert sequence(rotated) == whole[cut:] + whole[:cut]
        assert all(s < e for _c, _o, s, e in rotated)


def test_refine_cli_without_input_layout_gives_the_same_assembly(synthetic, tmp_path):
    """The refined assembly depends on the contact map and the contigs only: -i is optional."""
    common = ["-c", str(synthetic["cool"]), "-f", str(synthetic["fasta"]), "-w", "30000", "-l", "10000", "-j", "NNNNNN"]
    with_layout = CliRunner().invoke(refine_main, [*common, "-i", str(synthetic["layout"]), "-o", str(tmp_path / "a")])
    without = CliRunner().invoke(refine_main, [*common, "-o", str(tmp_path / "b"), "--origin-first"])
    assert with_layout.exit_code == 0, with_layout.output
    assert without.exit_code == 0, without.output
    assert "Input layout: none given" in without.output
    assert (tmp_path / "a" / "refined_genome.fa").read_text() == (tmp_path / "b" / "refined_genome.fa").read_text()


@pytest.mark.parametrize("flag", ["--qc-only", "--keep-unresolved"])
def test_refine_cli_layout_options_need_a_layout(synthetic, tmp_path, flag):
    result = CliRunner().invoke(
        refine_main, ["-c", str(synthetic["cool"]), "-f", str(synthetic["fasta"]), "-o", str(tmp_path / "o"), flag]
    )
    assert result.exit_code != 0
    assert "need an input layout" in result.output


# ---------------------------------------------------------------------------
# Second pass (--extend)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gapped(tmp_path_factory):
    """Linear chromosome whose 20 kb between ctg2 and ctg4 (ctg3) is missing from the assembly.

    The missing piece weakens the contacts right across the ctg2-ctg4 junction: too weak for
    the first pass (30 kb windows), enough for the second pass over whole scaffolds.
    """
    d = tmp_path_factory.mktemp("refine_gapped")
    rng = np.random.default_rng(1)
    cuts = [0, 60, 110, 170, 190, 250, 290, 330]
    genome_kb = cuts[-1]
    chrom = "".join(rng.choice(list("ACGT"), size=genome_kb * 1000))
    names = [f"ctg{k}" for k in range(len(cuts) - 1)]
    kept = [k for k in range(len(names)) if k != 3]
    fasta = d / "assembly.fa"
    fasta.write_text("".join(f">{names[k]}\n{chrom[cuts[k] * 1000 : cuts[k + 1] * 1000]}\n" for k in kept))
    contig_of = [max(i for i in range(len(names)) if cuts[i] <= f) for f in range(genome_kb)]
    frags = [f for f in range(genome_kb) if contig_of[f] != 3]
    bin_of = {f: b for b, f in enumerate(frags)}
    bins = pd.DataFrame(
        {
            "chrom": [names[contig_of[f]] for f in frags],
            "start": [(f - cuts[contig_of[f]]) * 1000 for f in frags],
            "end": [(f - cuts[contig_of[f]] + 1) * 1000 for f in frags],
        }
    )
    pos = np.array(frags) + 0.5
    lam = 40.0 / np.maximum(np.abs(pos[:, None] - pos[None, :]), 1.0) + 0.05
    counts = rng.poisson(np.triu(lam, 1))
    i, j = np.nonzero(counts)
    b1 = np.array([bin_of[frags[x]] for x in i])
    b2 = np.array([bin_of[frags[x]] for x in j])
    pixels = pd.DataFrame({"bin1_id": np.minimum(b1, b2), "bin2_id": np.maximum(b1, b2), "count": counts[i, j]})
    cool = d / "contacts.cool"
    cooler.create_cooler(
        str(cool),
        bins,
        pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True),
        ordered=True,
        symmetric_upper=True,
        dtypes={"count": np.int32},
    )
    lengths = {names[k]: (cuts[k + 1] - cuts[k]) * 1000 for k in kept}
    assembled = chrom[: cuts[3] * 1000] + chrom[cuts[4] * 1000 :]
    return {"dir": d, "fasta": fasta, "cool": cool, "lengths": lengths, "assembled": assembled, "kept": [names[k] for k in kept]}


def _refine_gapped(data, tmp_path, layout=None, **kwargs):
    path = None if layout is None else _write_layout(tmp_path / "layout.txt", layout, data["lengths"])
    kwargs = {"window": 30_000, "min_contig_length": 10_000, **kwargs}
    return run_refine(path, data["cool"], data["fasta"], tmp_path / "out", **kwargs)


def test_second_pass_joins_across_a_gap(gapped, tmp_path):
    first = _refine_gapped(gapped, tmp_path / "off", extend="off")
    assert first["scaffolds"] == 2 and first["extended_joins"] == 0
    summary = _refine_gapped(gapped, tmp_path / "on", extend="on")
    assert summary["extended_joins"] == 1
    records = list(SeqIO.parse(str(tmp_path / "on" / "out" / "refined_genome.fa"), "fasta"))
    assert len(records) == 1
    seq = str(records[0].seq)
    assert seq == gapped["assembled"] or _revcomp(seq) == gapped["assembled"]
    table = pd.read_csv(tmp_path / "on" / "out" / "junctions.tsv", sep="\t")
    extended = table[table["status"] == "extended"]
    assert len(extended) == 1
    assert {extended["end_a"].iloc[0], extended["end_b"].iloc[0]} == {"ctg2:R", "ctg4:L"}


TRUE_GAPPED = {"scaffold1": [("ctg0", 1), ("ctg1", 1), ("ctg2", 1), ("ctg4", 1), ("ctg5", 1), ("ctg6", 1)]}
SCRAMBLED_GAPPED = {"scaffold1": [("ctg0", -1), ("ctg4", 1), ("ctg1", -1), ("ctg5", 1), ("ctg2", -1), ("ctg6", -1)]}


@pytest.mark.parametrize(
    "layout,extended",
    [(TRUE_GAPPED, 1), (SCRAMBLED_GAPPED, 0), (None, 0)],
    ids=["layout-confirmed", "layout-contradicted", "no-layout"],
)
def test_extend_auto_trusts_long_range_contacts_only_when_the_layout_agrees(gapped, tmp_path, layout, extended):
    summary = _refine_gapped(gapped, tmp_path, layout=layout)  # extend="auto" by default
    assert summary["extended_joins"] == extended
    if layout is None:
        assert "needs an input layout" in summary["report"]
    elif extended:
        assert summary["agreement"] >= refine_module.EXTEND_MIN_AGREEMENT
    else:
        assert summary["agreement"] < refine_module.EXTEND_MIN_AGREEMENT
