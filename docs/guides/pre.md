# Preparing the contacts

instaGRAAL starts from the contigs and the Hi-C read pairs aligned on them.
`instagraal-pre` turns them into the contact map of the restriction fragments
of the contigs, in the two forms the next steps read.

```bash
instagraal-pre contigs.fa pairs.gz --enzyme DpnII,HinfI --output-dir pre
```

## The pairs

The pairs are a text file in the 4D Nucleome
[pairs format](../formats.md#pairs), plain or compressed with gzip or bgzip:
one line per read pair, with the contig and the position (1-based) of each
mate.

- `instagraal-pre` and `instagraal-post` read the columns named `chr1`,
  `pos1`, `chr2` and `pos2` in the `#columns` line, or else columns 2 to 5.
  Other columns are ignored.
- Pairs on sequences that are not in the FASTA file are left out.
- `instagraal-post` also reads the `#chromsize` lines, which give the contig
  lengths: without them, it skips the contig-level map (`<name>_contigs.cool`).

### With hicstuff

[hicstuff](https://github.com/koszullab/hicstuff) aligns the reads and stops
at the pairs with `--matrix none` (`-M none`, hicstuff 4.0.0 or later):

```bash
hicstuff pipeline -g contigs.fa -e DpnII,HinfI -m cutsite -M none -t 16 \
    -o hicstuff -P sample hic_R1.fq.gz hic_R2.fq.gz
```

It writes `hicstuff/sample.valid_idx.pairs.gz`, sorted and compressed, with
`#chromsize` lines. `-m cutsite` cuts the reads at the ligation sites of the
enzymes before aligning them, and `-m iterative` aligns them in growing
pieces: both align more reads than the default. `-D` removes PCR duplicates
(`sample.valid_idx_pcrfree.pairs.gz`).

### With bwa and pairtools

```bash
samtools faidx contigs.fa
cut -f 1,2 contigs.fa.fai > contigs.sizes
bwa index contigs.fa
bwa mem -5SP -t 16 contigs.fa hic_R1.fq.gz hic_R2.fq.gz \
    | pairtools parse --chroms-path contigs.sizes \
    | pairtools sort \
    | pairtools dedup -o pairs.gz
```

## The enzymes

`--enzyme` takes the restriction enzymes of the library, comma-separated, by
their names in [Biopython](https://biopython.org/docs/latest/api/Bio.Restriction.html):
`DpnII`, `MboI`, `HindIII`, `DpnII,HinfI` (the Arima kit)… Their sites are
merged, and the contigs cut at all of them into **restriction fragments**. A
name that Biopython does not know stops the run.

## What it writes

In `--output-dir` (the current directory by default):

| File | What it holds |
|---|---|
| `fragments_list.txt` | One line per restriction fragment: its contig, start, end, size and GC content |
| `info_contigs.txt` | One line per contig: its length and number of fragments |
| `abs_fragments_contacts_weighted.txt` | The contacts between fragments, as a sparse matrix |
| `<name>.cool` | The same contacts, as a cool file: what `instagraal-refine` reads |
| `<name>_hic_map.png` | A picture of the contact map, the contigs in the order of the FASTA file |

`<name>` is `--cool-name`, or the name of the pairs file without its
extensions (`yeast.pairs.gz` gives `yeast`). The first three files make the
input folder of the MCMC scaffolder, `instagraal`. [File formats](../formats.md#the-contacts-of-the-fragments)
describes them.

`instagraal-pre` first prints the statistics of the contigs, then each step:

```console
[1/5] Opening genome: yeast.contigs.fa.gz
[2/5] Digesting genome with enzyme(s): DpnII, HinfI
      → 75,032 restriction fragments
[3/5] Computing GC content per fragment
[4/5] Binning pairs into fragment contact matrix: yeast.pairs.gz
      → 1,335,066 valid pairs, 857,719 non-zero pixels
[5/5] Writing outputs to out/pre
```

A pair counts once, between the fragments of its two mates; a pair within one
fragment counts on the diagonal.

## Checking the contacts

`<name>_hic_map.png` shows the contacts between the contigs before
scaffolding: squares along the diagonal, one per contig, and spots between
contigs that will end up together. Few contacts between contigs, or a low
share of aligned reads, make for poor scaffolds: see
[Troubleshooting](../troubleshooting.md#poor-scaffolds).

## From hicstuff's graal matrix

`hicstuff pipeline -M graal`, without `--prefix`, writes the same three text
files as `instagraal-pre`, in its output directory, which `instagraal` reads
as its input folder. Refine needs the cool file of `instagraal-pre`, though:
the pipeline starts from the pairs.
