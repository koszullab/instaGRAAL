# Checking an assembly

A real assembly has no reference to compare with. instaGRAAL's outputs give
four checks of the scaffolds: their statistics, their contact map, the score
of each junction, and, for bacteria, their GC skew.

## Statistics

`instagraal-stats` prints the statistics of one or more FASTA files, side by
side; `instagraal-endtoend` ends with those of the contigs and the scaffolds.

```bash
instagraal-stats contigs.fa out/final_genome.fa --labels "Contigs,Scaffolds"
```

| Metric | What it is |
|---|---|
| Sequences | The number of contigs or scaffolds |
| Total length | Their total length, in bp: the scaffolds add the junctions (6 bp each by default) |
| Largest, Shortest, Mean, Median | Their lengths |
| N50, L50 | Half the assembly is in sequences of N50 bp or more, and in the L50 longest |
| N90, L90 | The same for 90% of the assembly |
| GC content | The share of G and C in the assembly |

A chromosome-level assembly has about as many large scaffolds as the
organism has chromosomes. Long scaffolds are not enough, though: a scaffold
that joins two chromosomes is long, and wrong.

## The contact map

`post/<name>_scaffolds_binned_<res>.png` shows the contacts of the scaffolds
(see [Contact maps of the assembly](post.md)):

- **Each scaffold is a square** on the diagonal, its contacts decaying away
  from the diagonal. Contacts between scaffolds are weaker, and even.
- **A bright square or stripe between two scaffolds**, at their ends, is a
  join that was missed.
- **A border inside a scaffold**, where the contacts stop, is a join that
  should not be: two chromosomes, or two parts of one in the wrong order.
- **A cross or an X** inside a bacterial chromosome comes from its two
  replication arms, which touch: it is not an error.

Centromeres (in yeast), telomeres and other loci that cluster in the nucleus
add dots between scaffolds, at the same places on each.

## The score of each junction

`refined/junctions.tsv` scores every join of the refined assembly, and
`instagraal-refine --qc-only` scores the joins of any layout: low scores, and
joins that refine contradicts, point to the junctions to check first (see
[Refining the scaffolds](refine.md#checking-a-layout)).

```bash
instagraal-refine --qc-only --info-frags final_info_frags.txt \
    --cool out/pre/sample.cool --fasta contigs.fa --output-dir qc
```

## GC skew

The two replication arms of a bacterial chromosome have opposite GC skews,
(G − C) / (G + C): positive from the origin to the terminus, negative after
it. `refine_report.txt` gives, for each scaffold of 500 kb or more, the share
of its length where the skew is positive, and how often it changes sign. A
whole, correctly ordered circular chromosome changes sign at most twice, and
is positive on about half its length; a folded or chimeric one changes sign
often, or stays mostly on one sign.

## The decay of contacts

`post/<name>_ps_curves.png` plots the probability of contact against the
distance, P(s), on the contigs and on the scaffolds. On the scaffolds, the
curve should go on decaying at distances the contigs did not reach. A tail
that flattens at long distances points to scaffolds that join pieces of
different chromosomes.

## With a reference

When a related genome exists, aligning the scaffolds on it shows the
remaining misjoins and inversions: as a dot plot, for instance with
[minimap2](https://github.com/lh3/minimap2) and
[D-GENIES](https://github.com/genotoul-bioinfo/dgenies).
