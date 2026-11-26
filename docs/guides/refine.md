# Refining the scaffolds

`instagraal-refine` joins whole contigs from the contacts next to their ends,
places the short ones, and closes and recentres circular chromosomes. It
needs no GPU. Given a layout, from the MCMC or any other scaffolder, it also
scores each of its joins: a check of the layout without a reference.

```bash
instagraal-refine --cool pre/yeast.cool --fasta contigs.fa --output-dir refined
instagraal-refine --info-frags polished/new_info_frags.txt --cool pre/yeast.cool \
    --fasta contigs.fa --output-dir refined
```

- `--cool` is the contact map of the restriction fragments that
  `instagraal-pre` wrote (see [Preparing the contacts](pre.md)).
- `--fasta` is the FASTA file of the contigs given to `instagraal-pre`.
- `--info-frags` is optional: a layout to check, such as the MCMC scaffolds
  (`new_info_frags.txt` of `instagraal-polish`, or the `info_frags.txt` of
  `instagraal`).

## Why refine

The MCMC's model of the contacts knows only their decay along a chromosome.
In many bacteria, the two replication arms of a chromosome are aligned: a
locus also touches its mirror locus on the other arm about as often as its
neighbours, which draws an X across the contact map. The MCMC explains those
contacts by folding the chromosome and interleaving its arms, and polishing
keeps the folded order. The map of such a scaffold looks smooth, with borders
where the arms switch.

Refine does not look at long-range contacts: it joins contigs from the
contacts next to their ends only, which the X does not change.

## How it joins contigs

1. The contig contacts are binned (`--binsize`, 5 kb) and balanced, and the
   decay of contacts with distance is measured inside the contigs, whose
   order is known.
2. For every pair of contig ends, the contacts between the 50 kb next to each
   end (`--window`, at most half of a contig) are compared, as a Poisson
   log-likelihood ratio, with what a direct junction would give and with the
   background between contigs.
3. Two ends are joined when each is the other's best partner, with a score of
   at least `--min-score` (10) and ahead of every other candidate by
   `--min-margin` (2). Scaffolds stay linear: no join closes a loop.
4. Contigs shorter than `--min-contig-length` (15 kb) carry too few contacts
   to be joined. Each is placed, whole, between the two contigs (or at the
   scaffold end) where its contacts fit best, when that place beats every
   other by `--short-margin` (10); the others become scaffolds of their own
   (`--no-place-short`: all of them).
5. A second pass (`--extend`) treats each scaffold of 100 kb or more as one
   sequence and joins them over 200 kb windows, which recovers joins too weak
   locally, across gaps or low coverage. Long-range contacts are those the X
   distorts, so `--extend auto` runs it only when refine confirms at least 30%
   of the joins of the `--info-frags` layout: a layout the X did not fold.
   `on` always runs it, and `off` never; without `--info-frags`, `auto` does
   not run it.

Higher `--min-score`, `--min-margin` and `--short-margin` give fewer, more
reliable joins. Refine keeps every contig whole: it cannot break a
misassembled contig.

## Circular chromosomes

A scaffold of at least 500 kb whose two outer ends would be joined, if loops
were allowed, is circular. It is cut at its replication origin, found from
its GC skew, so that it reads like a reference genome: its skew is positive
from its start to the terminus near its middle, and negative after it.

- `--terminus-first` cuts it at the terminus instead, which puts the origin in
  the middle of the scaffold and of its contact map.
- `--no-recenter` leaves circular scaffolds as they are joined.
- A scaffold whose skew does not show two clear replichores is left as it is.
  Skew cannot tell the two strands apart, so the strand is left as joined.

The report lists the circular scaffolds and where each was cut.

## Checking a layout

With `--info-frags`, refine scores every join between two contigs of 15 kb or
more in that layout, and gives each a verdict in `junctions.tsv`:

| Status | Meaning |
|---|---|
| `kept` | Refine makes the same join |
| `contradicted` | One of the two ends has a better partner |
| `unsupported` | Its score is below `--min-score` |
| `unresolved` | Its score reaches `--min-score`, but its ends found no confident partner |

`--qc-only` scores the layout and stops: it writes `junctions.tsv` (with no
status) and `refine_report.txt`, and no assembly. `--keep-unresolved` keeps
the `unresolved` joins of the layout in the refined assembly.

On the test data, refine checks the layout of the `legacy` pipeline (MCMC from
the bins, then polish) as follows:

```console
$ instagraal-refine -i new_info_frags.txt -c out/pre/yeast.cool -f yeast.contigs.fa.gz -o refined
...
Input layout: 106 joins between contigs >= 15,000 bp
  kept 86, contradicted 16 (an end has a better partner), unsupported 3 (score < 10.0), unresolved 1
Refined: 103 joins (17 new) + 3 from the second pass, 16 scaffolds, N50 924,431 bp (input N50 948,069 bp)
  second pass over whole scaffolds (200 kb windows): 3 joins (--extend auto: refine confirms 81% of the input layout's joins >= 30%)
...
GC skew of scaffolds >= 500,000 bp (a whole circular chromosome changes sign at most twice):
  input    3C-assembly|contig_15: 580,252 bp, skew > 0 on 71%, 8 sign changes
...
  input    3C-assembly|contig_22: 4,638,454 bp, skew > 0 on 56%, 74 sign changes
```

Here the layout has a 4.6 Mb scaffold, three times the longest yeast
chromosome, and refine contradicts 16 of its joins. The report also gives the
GC skew of the large scaffolds: a whole circular chromosome changes sign at
most twice, at its origin and terminus, and is positive on about half its
length; a folded one stays mostly on one sign.

## Outputs

| File | What it holds |
|---|---|
| `refined_genome.fa` | The refined assembly. `--junction` goes between contigs (none by default; `instagraal-endtoend` passes `NNNNNN`). |
| `refined_info_frags.txt` | Its layout, one line per contig ([info_frags](../formats.md#info_frags-files)). A recentred scaffold starts and ends with the two parts of the contig it is cut in. |
| `junctions.tsv` | Every join scored: those of the input layout, and those of the refined one (`joined`, `extended` by the second pass, `placed` short contigs, `closure` of circular scaffolds) |
| `refine_report.txt` | The summary that refine prints |

[File formats](../formats.md#junctionstsv) describes `junctions.tsv`.
`instagraal-post` makes the contact maps of the refined scaffolds from
`refined_info_frags.txt` (see [Contact maps of the assembly](post.md)).

## Memory

Refine scores every pair of contig ends at once: its memory grows with the
square of the number of contigs, to some 6 GB at 5,000 contigs.
`instagraal-endtoend` refuses to refine more than 5,000 contigs (see
[Running the pipeline](pipeline.md#more-than-5000-contigs)).
