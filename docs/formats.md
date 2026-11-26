# File formats

The files instaGRAAL reads and writes. Every table is tab-separated.

## Inputs

### Contigs

A FASTA file of the contigs, gzip-compressed (`.gz`) or not. A contig is read
by its name, the first word of its header; names with spaces or special
characters break the steps that match names between files.

### Pairs

The 4D Nucleome
[pairs format](https://github.com/4dn-dcic/pairix/blob/master/pairs_format_specification.md):
a header of lines starting with `#`, then one line per read pair, with the
contig and the position (1-based) of each mate. The pairs of the test data,
written by pairtools:

```text
## pairs format v1.0.0
#sorted: chr1-chr2-pos1-pos2
#shape: upper triangle
#chromsize: contig_001 17391
...
#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2 pair_type
SRR22130071.16970564	contig_001	1	contig_001	240	+	-	UU
```

- `instagraal-pre` and `instagraal-post` read the columns named `chr1`,
  `pos1`, `chr2` and `pos2` in the `#columns` line, or else columns 2 to 5,
  as here.
- `instagraal-post` reads the contig lengths from the `#chromsize` lines;
  without them, it skips `<name>_contigs.cool`.
- Gzip or bgzip compression is read from the name (`.gz`, `.bgz`).

## The contacts of the fragments

`instagraal-pre` writes the contact map of the restriction fragments as three
text files, the input folder of the MCMC scaffolder, and as a cool file. Each
text file starts with a header line, which instaGRAAL skips.

### `fragments_list.txt`

One line per restriction fragment, contig by contig, in the order of the
FASTA file. `id` starts again at 1 on each contig; `start_pos` and `end_pos`
are 0-based, the end excluded.

```text
id	chrom	start_pos	end_pos	size	gc_content
1	contig_001	0	786	786	0.15521628498727735
2	contig_001	786	943	157	0.2929936305732484
```

### `info_contigs.txt`

One line per contig: its length in bp, its number of fragments, and the
number of fragments of the contigs before it.

```text
contig	length	n_frags	cumul_length
contig_001	17391	30	0
contig_002	175172	1127	30
```

### `abs_fragments_contacts_weighted.txt`

The contact map, as a sparse matrix: after the header (the number of
fragments, twice, and the number of lines), one line per pair of fragments in
contact, as their 0-based indices in `fragments_list.txt` and the number of
read pairs between them.

```text
75032	75032	857719
0	0	283
0	1	314
```

hicstuff writes the same three files with `-M graal`.

### `<name>.cool`

The same contacts as a [cool file](https://github.com/open2c/cooler), one
bin per restriction fragment, counts in the upper triangle:
`instagraal-refine` reads it.

## `info_frags` files

A layout: the scaffolds, and the pieces of contig in each, in order.
`instagraal` writes `info_frags.txt`, `instagraal-polish`
`new_info_frags.txt`, `instagraal-refine` `refined_info_frags.txt`, and
`instagraal-endtoend` copies the last of them to `final_info_frags.txt`.

```text
>refined_scaffold_1
init_contig	id_frag	orientation	start	end
contig_108	107	1	0	116953
contig_104	103	1	0	82078
```

Each scaffold starts with its name (`>`) and a header line; each line after it
is a piece of the scaffold, in order:

| Column | What it is |
|---|---|
| `init_contig` | The contig the piece comes from |
| `id_frag` | The bin, in an MCMC layout; the contig's index in the FASTA file, in a refined one; `-1` for DNA that polishing put back |
| `orientation` | `1` as on the contig, `-1` reverse-complemented |
| `start`, `end` | The piece on the contig, 0-based, the end excluded |

The FASTA file of a layout holds the pieces in order, the junction (`NNNNNN`
by default) between pieces of different contigs.

## `junctions.tsv`

Every join that `instagraal-refine` scored, one per line:

```text
layout	scaffold	end_a	end_b	score	obs_exp	contacts	status
input	3C-assembly|contig_10	contig_029:R	contig_018:L	1786.84	1.472	468	kept
input	3C-assembly|contig_11	contig_053:R	contig_087:R	-39.25	0.185	33	contradicted
refined	refined_scaffold_1	contig_108:R	contig_104:L	1133.44	0.986	342	joined
```

| Column | What it is |
|---|---|
| `layout` | `input`, a join of `--info-frags`; `refined`, a join of the refined assembly |
| `scaffold` | The scaffold of the join |
| `end_a`, `end_b` | The two contig ends: `L` the start of the contig, `R` its end |
| `score` | The log-likelihood ratio of a junction against the background |
| `obs_exp` | The contacts between the two ends, over those a junction would give |
| `contacts` | The read pairs between the two ends |
| `status` | For `input`: `kept`, `contradicted`, `unsupported` or `unresolved` (empty with `--qc-only`). For `refined`: `joined`, `extended` (second pass), `placed` (short contig) or `closure` (the closing join of a circular scaffold) |

[Refining the scaffolds](guides/refine.md#checking-a-layout) explains the
statuses. `refine_report.txt` holds the summary that `instagraal-refine`
prints.

## The MCMC run

In `<output-dir>/<HIC_FOLDER name>/test_mcmc_<level>/`, `instagraal` writes,
after each cycle:

| File | What it holds |
|---|---|
| `genome.fasta` | The scaffolds |
| `info_frags.txt` | Their layout, bin by bin |
| `save_simu_step_<n>.txt` | The state of every bin after cycle `n`: position, start, scaffold and orientation |
| `matrix_cycle_<n>.png` | With `--save-matrix`: the contact map after cycle `n` |
| `list_likelihood.txt`, `list_n_contigs.txt`, `list_mean_len.txt`, `list_dist_init_genome.txt` | One value per move: the likelihood, the number of scaffolds, their mean length, the distance from the initial layout |
| `list_fact.txt`, `list_slope.txt`, `list_d.txt`, `list_d_max.txt`, `list_d_nuc.txt`, `list_success.txt` | The parameters of the contact model, as they are re-estimated |
| `list_mutations.txt` | The moves: the bin visited, its partner, and the move applied |

The scaffolds of the MCMC are named `3C-assembly|contig_<n>`. `pyramids/`, in
the output directory, holds the contact maps of every level:
`pyramid_1_no_thresh/` before the coverage filter, `pyramid_9_thresh_auto/`
after it, one `level_<n>/` folder each, and an HDF5 file of the contacts. In
the current directory, `instagraal` writes its log, `instagraal-<date>.log`,
and `sparsity_plot.pdf` and `sparsity_hist.pdf`, the share of fragments each
fragment is in contact with, which the coverage filter reads.

## The contact maps of the scaffolds

`instagraal-post` writes, with `<name>` the base name of the files:

| File | What it holds |
|---|---|
| `<name>_scaffolds_binned.mcool` | A multi-resolution cool file: the contacts of the scaffolds at each resolution, under `resolutions/<bp>`, with balancing weights |
| `<name>_scaffolds.cool` | One bin per scaffold |
| `<name>_contigs.cool` | One bin per contig, the contigs in the order of the scaffolds, from the pairs as they are |
| `<name>_lifted.pairs.gz` | The pairs, their positions on the scaffolds, with `#chromsize` lines for the scaffolds |
| `<name>_ps_curves.png` | P(s), the contact probability against the distance, on the contigs and on the scaffolds |
| `<name>_scaffolds_binned_<res>.png`, `<name>_scaffolds.png`, `<name>_contigs.png` | Pictures of the maps |

```text
#columns: readID chrom1 pos1 chrom2 pos2 strand1 strand2 pair_type
SRR22130071.16970564	refined_scaffold_15	53969	refined_scaffold_15	54208	+	-	UU
```
