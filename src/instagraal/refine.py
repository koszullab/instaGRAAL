"""Re-scaffold instaGRAAL assemblies from local Hi-C evidence at contig junctions.

Why this step exists
--------------------
instaGRAAL's MCMC scores a layout with a contact model that only decays with the
distance between two loci along the sequence. In many bacteria the two replication
arms of a chromosome are aligned, so a locus also contacts its mirror locus on the
other arm about as often as its neighbours (an X in the contact map). The model has
no term for that; it explains those contacts by folding the chromosome and
interleaving its arms, which gives a deceptively smooth map. ``instagraal-polish``
puts each input contig back together but keeps the folded contig order, so every
switch between arms shows up as a border in the final map.

What it does
------------
Contigs are kept whole. Every contig is binned (``binsize``) from the fragment-level
contact map written by ``instagraal-pre``; the matrix is ICE-balanced, and the decay
of contacts with distance is measured inside contigs, whose order is known. For every
pair of contig ends, only the contacts between the ``window`` next to each end (at
most half of the contig) are compared, as a Poisson log-likelihood ratio, with what a
direct junction would produce (that decay) and with the background between contigs.
Long-range contacts, which carry the inter-arm signal, play no part. Ends are then
joined greedily when they are each other's best partner, above ``min_score`` and ahead
of their next best candidate by ``min_margin``; cycles are not allowed, so scaffolds
stay linear.

Contigs shorter than ``min_contig_length`` carry too few contacts to be joined
reliably. Each is then placed, whole, into the slot of the new scaffolds (between two
long contigs, or at a scaffold end) whose ends its contacts fit best, when that slot
clearly beats every other one (``short_margin``); the others become scaffolds of their
own. Placement depends on the contact map only, not on where the input layout put
them: the scaffolder misplaces many short contigs.

The input layout (``new_info_frags.txt`` from ``instagraal-polish``, or the raw
``info_frags.txt``) is scored the same way, which gives a reference-free QC of its
joins, and GC skew is reported for large scaffolds: a correctly assembled circular
chromosome changes skew sign at most twice (origin, terminus) and spends about half
its length on each sign; a folded one stays mostly on one sign.
"""

import collections
import itertools
import math
import pathlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import cooler
import numpy as np
import scipy.sparse as sp

from ._scaffold_io import _parse_fasta, parse_info_frags, write_fasta

DEFAULT_BINSIZE = 5_000
DEFAULT_WINDOW = 50_000
DEFAULT_MIN_CONTIG_LENGTH = 15_000
DEFAULT_MIN_SCORE = 10.0
DEFAULT_MIN_MARGIN = 2.0
DEFAULT_SHORT_MARGIN = 10.0
REFINED_INFO_FRAGS_NAME = "refined_info_frags.txt"
REFINED_GENOME_NAME = "refined_genome.fa"
JUNCTIONS_NAME = "junctions.tsv"
REPORT_NAME = "refine_report.txt"
SKEW_MIN_SCAFFOLD = 500_000


# ---------------------------------------------------------------------------
# Contacts between contig bins
# ---------------------------------------------------------------------------


@dataclass
class ContigBins:
    """Contigs cut into fixed-size bins, with the symmetric contact matrix between bins."""

    names: list[str]
    lengths: dict[str, int]
    binsize: int
    offset: dict[str, int]
    nbins: dict[str, int]
    matrix: sp.csr_matrix
    contig_of_bin: np.ndarray = field(repr=False)


def load_contig_bins(cool_path: str | pathlib.Path, binsize: int = DEFAULT_BINSIZE) -> ContigBins:
    """Aggregate the fragment-level cool from instagraal-pre into contig bins.

    The diagonal (contacts within a bin) is dropped, as is usual before balancing.
    """
    clr = cooler.Cooler(str(cool_path))
    sizes = clr.chromsizes
    names = [str(c) for c in sizes.index]
    lengths = {c: int(sizes[c]) for c in names}
    nbins = {c: max(1, math.ceil(lengths[c] / binsize)) for c in names}
    offset, acc = {}, 0
    for c in names:
        offset[c] = acc
        acc += nbins[c]
    n = acc
    frags = clr.bins()[["chrom", "start", "end"]][:]
    mid = ((frags["start"].to_numpy() + frags["end"].to_numpy()) // 2).astype(np.int64)
    first = np.array([offset[c] for c in frags["chrom"].astype(str)], dtype=np.int64)
    last = first + np.array([nbins[c] - 1 for c in frags["chrom"].astype(str)], dtype=np.int64)
    frag_bin = np.minimum(first + mid // binsize, last)
    pix = clr.pixels()[:]
    i = frag_bin[pix["bin1_id"].to_numpy()]
    j = frag_bin[pix["bin2_id"].to_numpy()]
    v = pix["count"].to_numpy().astype(float)
    upper = sp.coo_matrix((v, (i, j)), shape=(n, n)).tocsr()
    sym = (upper + upper.T).tolil()
    sym.setdiag(0)
    matrix = sym.tocsr()
    matrix.eliminate_zeros()
    contig_of_bin = np.repeat(np.arange(len(names)), [nbins[c] for c in names])
    return ContigBins(names, lengths, binsize, offset, nbins, matrix, contig_of_bin)


def ice_weights(
    matrix: sp.csr_matrix, min_coverage_frac: float = 0.05, max_iter: int = 500, tol: float = 1e-4
) -> tuple[np.ndarray, np.ndarray]:
    """Iterative correction: weights w so that ``matrix * w_i * w_j`` has equal row sums.

    Bins whose raw coverage is below ``min_coverage_frac`` of the median (non-zero)
    coverage are masked (weight 0). Returns ``(weights, mask)``, with weights scaled
    so that balanced rows sum to 1.
    """
    cov = np.asarray(matrix.sum(axis=1)).ravel()
    nonzero = cov[cov > 0]
    if nonzero.size == 0:
        raise ValueError("The contact map is empty.")
    mask = cov < max(1.0, min_coverage_frac * float(np.median(nonzero)))
    w = np.where(mask, 0.0, 1.0)
    for _ in range(max_iter):
        marg = w * (matrix @ w)
        mean = marg[~mask].mean()
        w[~mask] /= marg[~mask] / mean
        if np.std(marg[~mask] / mean) < tol:
            break
    marg = w * (matrix @ w)
    w /= math.sqrt(marg[~mask].mean())
    return w, mask


def contact_decay(cb: ContigBins, w: np.ndarray, mask: np.ndarray, kmax: int) -> tuple[np.ndarray, float]:
    """Mean balanced contacts between bins ``k`` apart inside contigs, for k = 0..kmax.

    Distances with too few bin pairs (contigs shorter than k bins) are filled with a
    power law fitted on the measured ones. Returns ``(E, slope)``.
    """
    sums, counts = np.zeros(kmax + 1), np.zeros(kmax + 1)
    for c in cb.names:
        n = cb.nbins[c]
        if n < 2:
            continue
        idx = np.arange(cb.offset[c], cb.offset[c] + n)
        block = cb.matrix[idx][:, idx].toarray() * np.outer(w[idx], w[idx])
        ok = ~mask[idx]
        for k in range(1, min(n, kmax + 1)):
            valid = ok[:-k] & ok[k:]
            sums[k] += np.diagonal(block, k)[valid].sum()
            counts[k] += valid.sum()
    ks = np.arange(kmax + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        measured = np.where(counts > 0, sums / counts, np.nan)
    fit = (ks >= 1) & (counts >= 20) & np.isfinite(measured) & (measured > 0)
    if fit.sum() < 2:
        raise ValueError("Not enough contacts inside contigs to measure how contacts decay with distance.")
    slope, intercept = np.polyfit(np.log(ks[fit]), np.log(measured[fit]), 1)
    decay = np.where(fit, measured, np.exp(intercept) * np.maximum(ks, 1).astype(float) ** slope)
    return decay, float(slope)


def background_level(cb: ContigBins, w: np.ndarray, mask: np.ndarray) -> float:
    """Mean balanced contacts between bins of different contigs (zeros included)."""
    coo = cb.matrix.tocoo()
    inter = cb.contig_of_bin[coo.row] != cb.contig_of_bin[coo.col]
    total = float(np.sum(coo.data[inter] * w[coo.row[inter]] * w[coo.col[inter]]))
    ok_per_contig = np.bincount(cb.contig_of_bin[~mask], minlength=len(cb.names)).astype(float)
    n_ok = ok_per_contig.sum()
    pairs = n_ok * n_ok - float(np.sum(ok_per_contig**2))
    return total / pairs if pairs > 0 else 0.0


# ---------------------------------------------------------------------------
# Scoring contig-end junctions
# ---------------------------------------------------------------------------

LEFT, RIGHT = "L", "R"
End = tuple[str, str]  # (contig, side)
Join = tuple[End, End]


@dataclass
class EndScores:
    """Scores for every pair of contig ends (``ends[k] = (contig, side)``)."""

    ends: list[tuple[str, str]]
    index: dict[tuple[str, str], int]
    score: np.ndarray  # log-likelihood ratio, junction vs background
    oe: np.ndarray  # observed / expected for a junction
    contacts: np.ndarray  # raw contacts in the two windows
    decay_slope: float
    background: float


def end_bins(cb: ContigBins, contig: str, side: str, window_bins: int) -> np.ndarray:
    """Bins of the ``window_bins`` closest to one contig end, nearest first.

    Each end gets at most half of its contig's bins, so that the two ends of a contig
    shorter than two windows do not share bins: shared bins would make them look
    alike and leave the contig's orientation to chance.
    """
    n, o = cb.nbins[contig], cb.offset[contig]
    m = min(window_bins, max(1, n // 2))
    return np.arange(o, o + m) if side == LEFT else np.arange(o + n - 1, o + n - 1 - m, -1)


def score_ends(
    cb: ContigBins,
    contigs: list[str],
    window: int = DEFAULT_WINDOW,
) -> EndScores:
    """Score every pair of ends of ``contigs`` as a potential direct junction.

    For ends A and B, with ``d_a``/``d_b`` the bin distances from each end, a direct
    junction predicts ``E(d_a + d_b + 1)`` balanced contacts per bin pair, the
    background predicts the mean between contigs. The score is the Poisson
    log-likelihood ratio of the observed counts under the two, so it weighs both the
    level and the amount of data; ``oe`` is observed / expected-for-a-junction.
    """
    window_bins = max(1, window // cb.binsize)
    w, mask = ice_weights(cb.matrix)
    decay, slope = contact_decay(cb, w, mask, 2 * window_bins + 1)
    bg = background_level(cb, w, mask)
    if bg <= 0:
        raise ValueError("No contacts between contigs: nothing to scaffold from.")
    ends = [(c, s) for c in contigs for s in (LEFT, RIGHT)]
    index = {e: k for k, e in enumerate(ends)}
    n_ends = len(ends)
    inv = np.zeros((n_ends, window_bins))  # 1 / weight of each end bin (0 when masked)
    members: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
    for k, (c, s) in enumerate(ends):
        bins = end_bins(cb, c, s, window_bins)
        for d, g in enumerate(bins):
            if not mask[g]:
                inv[k, d] = 1.0 / w[g]
                members[int(g)].append((k, d))
    dist = np.add.outer(np.arange(window_bins), np.arange(window_bins)) + 1
    expected_join = inv @ decay[dist] @ inv.T  # sum of expected raw counts for a junction
    expected_bg = bg * np.outer(inv.sum(1), inv.sum(1))
    log_ratio = np.log(np.maximum(decay, 1e-300) / bg)

    # observed contacts between the end windows, accumulated from the sparse matrix
    used = np.array(sorted(members), dtype=np.int64)
    sub = cb.matrix[used][:, used].tocoo()
    obs = np.zeros((n_ends, n_ends))
    obs_log = np.zeros((n_ends, n_ends))
    layer_end = np.full((2, used.size), -1, dtype=np.int64)
    layer_d = np.zeros((2, used.size), dtype=np.int64)
    for u, g in enumerate(used):
        for layer, (k, d) in enumerate(members[int(g)][:2]):
            layer_end[layer, u], layer_d[layer, u] = k, d
    for la in range(2):
        for lb in range(2):
            ea, eb = layer_end[la, sub.row], layer_end[lb, sub.col]
            ok = (ea >= 0) & (eb >= 0)
            ea, eb, val = ea[ok], eb[ok], sub.data[ok]
            k_dist = layer_d[la, sub.row[ok]] + layer_d[lb, sub.col[ok]] + 1
            np.add.at(obs, (ea, eb), val)
            np.add.at(obs_log, (ea, eb), val * log_ratio[k_dist])

    score = obs_log - (expected_join - expected_bg)
    with np.errstate(divide="ignore", invalid="ignore"):
        oe = np.where(expected_join > 0, obs / expected_join, 0.0)
    contig_ids = np.repeat(np.arange(len(contigs)), 2)
    same = contig_ids[:, None] == contig_ids[None, :]
    score[same] = -np.inf
    oe[same] = 0.0
    return EndScores(ends, index, score, oe, obs, slope, bg)


# ---------------------------------------------------------------------------
# Joining ends into scaffolds
# ---------------------------------------------------------------------------


class _Components:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)


def join_ends(
    es: EndScores,
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    extra: list[Join] | None = None,
) -> list[Join]:
    """Join ends that are each other's best free partner, with ``score >= min_score``
    and a lead of at least ``min_margin`` over each end's next best candidate.

    Repeats until no such pair is left; never closes a cycle. Joins listed in
    ``extra`` (e.g. the input layout's) are then added where both ends are still
    free, the score reaches ``min_score`` and no cycle is formed.
    """
    n = len(es.ends)
    contig_of = [e[0] for e in es.ends]
    comps = _Components(set(contig_of))
    free = np.ones(n, dtype=bool)
    joins = []
    while True:
        comp = np.array([comps.find(c) for c in contig_of])
        allowed = free[:, None] & free[None, :] & (comp[:, None] != comp[None, :])
        sc = np.where(allowed, es.score, -np.inf)
        if not np.isfinite(sc).any():
            break
        best = np.argmax(sc, axis=1)
        best_score = sc[np.arange(n), best]
        runner_up = sc.copy()
        runner_up[np.arange(n), best] = -np.inf
        second = np.maximum(runner_up.max(axis=1), 0.0)
        margin = best_score - second
        accepted = []
        for a in np.nonzero(free)[0]:
            b = best[a]
            if a < b and best[b] == a and best_score[a] >= min_score and margin[a] >= min_margin and margin[b] >= min_margin:
                accepted.append((best_score[a], a, b))
        if not accepted:
            break
        for _s, a, b in sorted(accepted, reverse=True):
            if free[a] and free[b] and comps.find(contig_of[a]) != comps.find(contig_of[b]):
                free[a] = free[b] = False
                comps.union(contig_of[a], contig_of[b])
                joins.append((es.ends[a], es.ends[b]))
    for ea, eb in extra or []:
        ia, ib = es.index.get(ea), es.index.get(eb)
        if ia is None or ib is None or not (free[ia] and free[ib]):
            continue
        if es.score[ia, ib] >= min_score and comps.find(contig_of[ia]) != comps.find(contig_of[ib]):
            free[ia] = free[ib] = False
            comps.union(contig_of[ia], contig_of[ib])
            joins.append((ea, eb))
    return joins


def build_scaffolds(contigs: list[str], joins: list[Join]) -> list[list[tuple[str, int]]]:
    """Chains of ``(contig, orientation)`` from end-to-end joins (no cycles)."""
    partner: dict[End, End] = {}
    for ea, eb in joins:
        partner[ea], partner[eb] = eb, ea
    seen: set[str] = set()
    scaffolds: list[list[tuple[str, int]]] = []
    for start in contigs:
        if start in seen:
            continue
        # 1. walk out through left ends until reaching an end with no partner
        c, side, visited = start, LEFT, {start}
        while (c, side) in partner:
            nxt_c, nxt_side = partner[(c, side)]
            if nxt_c in visited:
                break
            visited.add(nxt_c)
            c, side = nxt_c, RIGHT if nxt_side == LEFT else LEFT
        # 2. read the chain from that free end
        entry, chain = (c, side), []
        while True:
            cc, entry_side = entry
            seen.add(cc)
            chain.append((cc, 1 if entry_side == LEFT else -1))
            nxt = partner.get((cc, RIGHT if entry_side == LEFT else LEFT))
            if nxt is None or nxt[0] in seen:
                break
            entry = nxt
        scaffolds.append(chain)
    return scaffolds


def subset_scores(es: EndScores, contigs: list[str]) -> EndScores:
    """The scores between the ends of ``contigs`` only."""
    ends = [(c, s) for c in contigs for s in (LEFT, RIGHT)]
    idx = np.array([es.index[e] for e in ends], dtype=np.int64)
    grid = np.ix_(idx, idx)
    return EndScores(
        ends, {e: k for k, e in enumerate(ends)}, es.score[grid], es.oe[grid], es.contacts[grid], es.decay_slope, es.background
    )


def place_short_contigs(
    chains: list[list[tuple[str, int]]],
    es: EndScores,
    candidates: Iterable[str],
    anchors: set[str],
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_SHORT_MARGIN,
) -> tuple[list[list[tuple[str, int]]], dict[str, float]]:
    """Insert contigs too short to be joined into the chains, from their contacts.

    A slot lies between two consecutive ``anchors`` contigs of a chain, or at either end
    of a chain. Contig S placed in orientation o between the exit end ``a`` of the
    contig before and the entry end ``b`` of the contig after scores
    ``score[a, S_in] + score[S_out, b]`` (one term at a chain end). S goes to its best
    slot if that scores at least ``min_score`` and leads the best other slot by
    ``min_margin``; within a slot it sits next to the contig it is more attached to.
    Returns the new chains and the score of each placed contig.
    """

    def exit_end(c: str, o: int) -> End:
        return (c, RIGHT if o == 1 else LEFT)

    def entry_end(c: str, o: int) -> End:
        return (c, LEFT if o == 1 else RIGHT)

    def sc(e1: End | None, e2: End | None) -> float:
        return 0.0 if e1 is None or e2 is None else float(es.score[es.index[e1], es.index[e2]])

    slots = []  # (chain, position of the anchor before or None, position of the anchor after or None)
    for ci, chain in enumerate(chains):
        pos = [k for k, (c, _o) in enumerate(chain) if c in anchors]
        if pos:
            slots += [(ci, None, pos[0]), *((ci, k1, k2) for k1, k2 in itertools.pairwise(pos)), (ci, pos[-1], None)]
    chosen: dict[tuple, list[tuple[float, str, int]]] = collections.defaultdict(list)
    placed: dict[str, float] = {}
    for s in candidates:
        if (s, LEFT) not in es.index:
            continue
        options = []
        for ci, k1, k2 in slots:
            a = exit_end(*chains[ci][k1]) if k1 is not None else None
            b = entry_end(*chains[ci][k2]) if k2 is not None else None
            # (total, orientation, attachment to the contig before minus attachment to the one after)
            best = max(
                (
                    (sc(a, entry_end(s, o)) + sc(exit_end(s, o), b), o, sc(a, entry_end(s, o)) - sc(exit_end(s, o), b))
                    for o in (1, -1)
                ),
                key=lambda x: x[0],
            )
            options.append((best[0], (ci, k1, k2), best[1], best[2]))
        if not options:
            break
        options.sort(key=lambda x: -x[0])
        second = options[1][0] if len(options) > 1 else 0.0
        top, slot, ori, lean = options[0]
        if top >= min_score and top - max(second, 0.0) >= min_margin:
            chosen[slot].append((lean, s, ori))
            placed[s] = top
    out = []
    for ci, chain in enumerate(chains):
        new = list(chain)
        # fill the slots from the end of the chain backwards, so that positions stay valid
        for (cj, k1, k2), items in sorted(chosen.items(), key=lambda kv: -(len(chain) if kv[0][2] is None else kv[0][2])):
            if cj != ci:
                continue
            items.sort(key=lambda x: -x[0])  # most attached to the contig before first
            near_a = [(s, o) for lean, s, o in items if lean >= 0]
            near_b = [(s, o) for lean, s, o in items if lean < 0]
            if k1 is None:
                new = near_a + near_b + new
            elif k2 is None:
                new = new + near_a + near_b
            else:
                new = new[: k1 + 1] + near_a + new[k1 + 1 : k2] + near_b + new[k2:]
        out.append(new)
    return out, placed


# ---------------------------------------------------------------------------
# Input layout, GC skew, outputs
# ---------------------------------------------------------------------------


def layout_contig_joins(scaffolds: dict[str, list], keep: set[str]) -> list[tuple[str, End, End]]:
    """Joins between consecutive contigs (those in ``keep``) of an info_frags layout.

    Consecutive bins of one contig form a block whose orientation is the
    length-weighted majority of its bins.
    """
    joins = []
    for name, bins in scaffolds.items():
        blocks: list[list] = []
        for contig, _id, start, end, ori in bins:
            if contig not in keep:
                continue
            if blocks and blocks[-1][0] == contig:
                blocks[-1][1] += ori * (end - start)
            else:
                blocks.append([contig, ori * (end - start)])
        for (ca, oa), (cb, ob) in itertools.pairwise(blocks):
            if ca != cb:
                joins.append((name, (ca, RIGHT if oa >= 0 else LEFT), (cb, LEFT if ob >= 0 else RIGHT)))
    return joins


_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def gc_skew_summary(seq: str, window: int = 10_000, smooth: int = 10) -> tuple[float, int]:
    """Share of the sequence with positive (smoothed) GC skew, and number of sign changes."""
    arr = np.frombuffer(seq.upper().encode(), dtype=np.uint8)
    n = len(arr) // window
    if n < smooth:
        return float("nan"), 0
    g = (arr[: n * window] == ord("G")).reshape(n, window).sum(1)
    c = (arr[: n * window] == ord("C")).reshape(n, window).sum(1)
    skew = (g - c) / np.maximum(g + c, 1)
    sm = np.convolve(skew, np.ones(smooth) / smooth, mode="same")
    signs = np.sign(sm[sm != 0])
    return float((sm > 0).mean()), int(np.sum(signs[1:] != signs[:-1]))


def _scaffold_sequence(genome: dict[str, str], pieces: Iterable[tuple[str, int, int, int]]) -> str:
    out = []
    for contig, ori, start, end in pieces:
        s = genome[contig][start:end]
        out.append(s if ori == 1 else s.translate(_RC)[::-1])
    return "".join(out)


def _n50(sizes: Iterable[int]) -> int:
    sizes = sorted(sizes, reverse=True)
    half, acc = sum(sizes) / 2, 0
    for s in sizes:
        acc += s
        if acc >= half:
            return int(s)
    return 0


def run_refine(
    info_frags: str | pathlib.Path,
    cool_path: str | pathlib.Path,
    fasta: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    binsize: int = DEFAULT_BINSIZE,
    window: int = DEFAULT_WINDOW,
    min_contig_length: int = DEFAULT_MIN_CONTIG_LENGTH,
    min_score: float = DEFAULT_MIN_SCORE,
    min_margin: float = DEFAULT_MIN_MARGIN,
    junction: str = "",
    keep_unresolved: bool = False,
    qc_only: bool = False,
    place_short: bool = True,
    short_margin: float = DEFAULT_SHORT_MARGIN,
) -> dict[str, Any]:
    """Score the input layout's joins and, unless ``qc_only``, re-scaffold the contigs.

    Writes ``junctions.tsv`` and ``refine_report.txt`` and, unless ``qc_only``,
    ``refined_info_frags.txt`` and ``refined_genome.fa`` (whole contigs, joined with
    ``junction`` between them) into ``output_dir``. Returns a summary dictionary.
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genome = {r.id: str(r.seq) for r in _parse_fasta(str(fasta))}
    cb = load_contig_bins(cool_path, binsize)
    missing = [c for c in genome if c not in cb.lengths]
    if missing:
        raise ValueError(f"{len(missing)} contig(s) of the FASTA are absent from the contact map, e.g. {missing[0]}.")
    layout = parse_info_frags(str(info_frags))
    unknown = {b[0] for bins in layout.values() for b in bins} - set(genome)
    if unknown:
        raise ValueError(f"The layout uses contigs absent from the FASTA, e.g. {min(unknown)}.")

    contigs = [c for c in genome if len(genome[c]) >= min_contig_length]
    short = [c for c in genome if len(genome[c]) < min_contig_length]
    es_all = score_ends(cb, contigs + short, window)
    es = subset_scores(es_all, contigs)
    keep = set(contigs)
    input_joins = layout_contig_joins(layout, keep)

    refined_joins: list[Join] = []
    if not qc_only:
        extra = [(ea, eb) for _name, ea, eb in input_joins] if keep_unresolved else None
        refined_joins = join_ends(es, min_score, min_margin, extra)
    refined_set = {frozenset(j) for j in refined_joins}
    partner: dict[End, End] = {}
    for ea, eb in refined_joins:
        partner[ea], partner[eb] = eb, ea

    def status(ea: End, eb: End) -> str:
        if frozenset([ea, eb]) in refined_set:
            return "kept"
        if partner.get(ea, eb) != eb or partner.get(eb, ea) != ea:
            return "contradicted"
        if es.score[es.index[ea], es.index[eb]] < min_score:
            return "unsupported"
        return "unresolved"

    rows = []
    for name, ea, eb in input_joins:
        a, b = es.index[ea], es.index[eb]
        rows.append(("input", name, ea, eb, es.score[a, b], es.oe[a, b], es.contacts[a, b], "" if qc_only else status(ea, eb)))

    statuses = collections.Counter(str(r[-1]) for r in rows)
    summary: dict[str, Any] = {
        "contigs": len(genome),
        "scaffolded_contigs": len(contigs),
        "decay_slope": es.decay_slope,
        "input_joins": len(input_joins),
        "input_statuses": statuses,
    }

    # GC skew of the large input scaffolds
    skew_lines = []
    for name, bins in layout.items():
        seq = _scaffold_sequence(genome, [(c, o, s, e) for c, _i, s, e, o in bins])
        if len(seq) >= SKEW_MIN_SCAFFOLD:
            share, changes = gc_skew_summary(seq)
            skew_lines.append(f"  input    {name}: {len(seq):,} bp, skew > 0 on {share:.0%}, {changes} sign changes")

    if not qc_only:
        chains = build_scaffolds(contigs, refined_joins)
        placed: dict[str, float] = {}
        if place_short:
            chains, placed = place_short_contigs(chains, es_all, short, keep, min_score, short_margin)
        chains += [[(c, 1)] for c in short if c not in placed]
        chains.sort(key=lambda ch: -sum(len(genome[c]) for c, _o in ch))
        fasta_index = {c: k for k, c in enumerate(genome)}
        info_path = output_dir / REFINED_INFO_FRAGS_NAME
        with open(info_path, "w") as fh:
            for k, chain in enumerate(chains, start=1):
                fh.write(f">refined_scaffold_{k}\ninit_contig\tid_frag\torientation\tstart\tend\n")
                fh.writelines(f"{c}\t{fasta_index[c]}\t{o}\t0\t{len(genome[c])}\n" for c, o in chain)
        write_fasta(str(fasta), str(info_path), output=str(output_dir / REFINED_GENOME_NAME), junction=junction)
        for k, chain in enumerate(chains, start=1):
            name = f"refined_scaffold_{k}"
            # the joins between long contigs, then the neighbours of each placed short contig
            long_pairs = itertools.pairwise([(c, o) for c, o in chain if c in keep])
            short_pairs = ((x, y) for x, y in itertools.pairwise(chain) if x[0] not in keep or y[0] not in keep)
            for verdict, pairs in (("joined", long_pairs), ("placed", short_pairs)):
                for (ca, oa), (cbn, ob) in pairs:
                    ea = (ca, RIGHT if oa == 1 else LEFT)
                    eb = (cbn, LEFT if ob == 1 else RIGHT)
                    a, b = es_all.index[ea], es_all.index[eb]
                    rows.append(("refined", name, ea, eb, es_all.score[a, b], es_all.oe[a, b], es_all.contacts[a, b], verdict))
            seq = _scaffold_sequence(genome, [(c, o, 0, len(genome[c])) for c, o in chain])
            if len(seq) >= SKEW_MIN_SCAFFOLD:
                share, changes = gc_skew_summary(seq)
                skew_lines.append(f"  refined  {name}: {len(seq):,} bp, skew > 0 on {share:.0%}, {changes} sign changes")
        sizes = [sum(len(genome[c]) for c, _o in ch) for ch in chains]
        summary.update(
            refined_joins=len(refined_joins),
            new_joins=len(refined_joins) - statuses.get("kept", 0),
            scaffolds=len(chains),
            scaffold_n50=_n50(sizes),
            input_n50=_n50([sum(e - s for _c, _i, s, e, _o in bins) for bins in layout.values()]),
            short_contigs=len(short),
            short_placed=len(placed),
            short_placed_bp=sum(len(genome[c]) for c in placed),
            short_alone_bp=sum(len(genome[c]) for c in short if c not in placed),
        )

    with open(output_dir / JUNCTIONS_NAME, "w") as fh:
        fh.write("layout\tscaffold\tend_a\tend_b\tscore\tobs_exp\tcontacts\tstatus\n")
        for lay, name, ea, eb, sc, oe, n, verdict in rows:
            fh.write(f"{lay}\t{name}\t{ea[0]}:{ea[1]}\t{eb[0]}:{eb[1]}\t{sc:.2f}\t{oe:.3f}\t{n:.0f}\t{verdict}\n")

    lines = [
        "instagraal-refine",
        (
            f"  binsize {binsize:,} bp, window {window:,} bp, contigs >= {min_contig_length:,} bp: "
            f"{len(contigs)} of {len(genome)}; min score {min_score}, min margin {min_margin}"
        ),
        f"  contact decay exponent inside contigs: {es.decay_slope:.2f}",
        f"Input layout: {len(input_joins)} joins between contigs >= {min_contig_length:,} bp",
    ]
    if not qc_only:
        lines.append(
            f"  kept {statuses.get('kept', 0)}, contradicted {statuses.get('contradicted', 0)} (an end has a better partner), "
            f"unsupported {statuses.get('unsupported', 0)} (score < {min_score}), unresolved {statuses.get('unresolved', 0)}"
        )
        lines.append(
            f"Refined: {summary['refined_joins']} joins ({summary['new_joins']} new), {summary['scaffolds']} scaffolds, "
            f"N50 {summary['scaffold_n50']:,} bp (input N50 {summary['input_n50']:,} bp)"
        )
        if summary["short_contigs"]:
            lines.append(
                f"  contigs < {min_contig_length:,} bp: {summary['short_placed']} of {summary['short_contigs']} "
                f"({summary['short_placed_bp']:,} bp) placed from their contacts (margin {short_margin}); "
                f"the others ({summary['short_alone_bp']:,} bp) are scaffolds of their own"
                if place_short
                else f"  contigs < {min_contig_length:,} bp: {summary['short_contigs']} ({summary['short_alone_bp']:,} bp), "
                "each a scaffold of its own"
            )
    else:
        supported = sum(1 for r in rows if r[4] >= min_score)
        lines.append(f"  {supported} reach the minimum score of {min_score}")
    if skew_lines:
        lines.append(
            f"GC skew of scaffolds >= {SKEW_MIN_SCAFFOLD:,} bp (a whole circular chromosome changes sign at most twice):"
        )
        lines += skew_lines
    report = "\n".join(lines)
    (output_dir / REPORT_NAME).write_text(report + "\n")
    print(report)
    summary["report"] = report
    return summary
