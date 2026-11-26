# CLI reference

instaGRAAL installs one command per step, and `instagraal-endtoend`, which
runs them in turn. This page documents every command and option; `-h/--help`
prints the same in a terminal.

```console
instagraal-endtoend --help
instagraal-endtoend --version
```

| Command | What it does |
|---|---|
| [`instagraal-endtoend`](#instagraal-endtoend) | Runs the pipeline: the steps below, as `--mode` says. |
| [`instagraal-pre`](#instagraal-pre) | Makes the contact map of the restriction fragments, from the contigs and the pairs. |
| [`instagraal`](#instagraal) | Scaffolds the bins of the contigs with the MCMC, on a GPU. |
| [`instagraal-polish`](#instagraal-polish) | Polishes a layout of the MCMC. |
| [`instagraal-refine`](#instagraal-refine) | Joins whole contigs from the contacts next to their ends, and scores the joins of a layout. |
| [`instagraal-post`](#instagraal-post) | Lifts the pairs over to the scaffolds, and writes their contact maps. |
| [`instagraal-stats`](#instagraal-stats) | Prints the statistics of assemblies, side by side. |
| [`instagraal-test`](#instagraal-test) | Checks a GPU installation on a test dataset. |

New to instaGRAAL? Start with the [Quickstart](quickstart.md) or
[The end-to-end pipeline](end-to-end.md); for how it works, see
[Concepts](concepts.md).

## Conventions

| Convention | Meaning |
|---|---|
| Compressed files | FASTA and pairs files can be gzip-compressed (`.gz`; pairs also `.bgz`), as their name says. |
| Output directories | A command writes into its output directory, created if missing (`instagraal` creates only the last directory of the path), and replaces the files it writes there. `instagraal` reuses the pyramid it finds there (see [Scaffolding on a GPU](guides/scaffolding.md#the-pyramid)). |
| Enzymes | Restriction enzymes by their names in Biopython, comma-separated: `DpnII,HinfI`. |
| Layouts | The `info_frags` files that `instagraal`, `instagraal-polish`, `instagraal-refine` and `instagraal-endtoend` write, and that the next steps read (see [File formats](formats.md#info_frags-files)). |
| Junctions | The sequence between two contigs in a scaffold: `NNNNNN` by default with `instagraal-polish` and `instagraal-endtoend`, none by default with `instagraal-refine`, none in the MCMC's `genome.fasta`. `instagraal-post --junction` takes its length (see [Contact maps of the assembly](guides/post.md#the-junction)). |
| Logs | `instagraal` writes its log, `instagraal-<date>.log`, in the current directory. The other commands print to the terminal. |
| Random numbers | The MCMC visits the bins in a random order: two runs of `instagraal` give close, not identical, layouts. The other steps give the same output for the same input. |
| `-V, --version` | Prints the version: `instagraal`, `instagraal-endtoend`, `instagraal-refine`, `instagraal-stats` and `instagraal-test`. |

## Pipeline

### `instagraal-endtoend` { #instagraal-endtoend }

Run the pipeline: `instagraal-pre`, then the steps of `--mode`, then
`instagraal-post` and `instagraal-stats`. **Guide:**
[Running the pipeline](guides/pipeline.md).

```console
instagraal-endtoend [OPTIONS] FASTA PAIRS
```

- **Inputs**: `FASTA`, the contigs, and `PAIRS`, the Hi-C pairs aligned on
  them ([pairs format](formats.md#pairs)).
- **Modes**: a preset, `default` (`mcmc,refine`, needs a GPU), `cpu`
  (`refine,extend`) or `legacy` (`bomb,mcmc,polish`), or steps separated by
  commas, in any order: `mcmc`, `bomb`, `polish`, `refine`, `extend`,
  `noextend` (see [Modes](guides/pipeline.md#modes)). The GPU is checked only
  when the mode has `mcmc`.
- **More than 5,000 contigs**: without `--mode`, the run falls back to
  `legacy`; a mode with `refine` is refused.
- **Outputs**: in `--output-dir`, one folder per step (`pre/`, `mcmc/`,
  `polished/`, `refined/`, `post/`), and the final assembly as
  `final_genome.fa` and `final_info_frags.txt`, whatever the mode (see
  [Outputs](guides/pipeline.md#outputs)). The run ends with the commands it
  ran.

| Option | Default | Description |
|---|---|---|
| `-e, --enzyme` | required | The restriction enzymes of the library, comma-separated. |
| `-o, --output-dir` | `out` | The directory of every output, created if missing. |
| `-m, --mode` | `default` | The steps to run between `instagraal-pre` and `instagraal-post`: a preset or steps, as above. |
| `--device` | `0` | The GPU to run the MCMC on, numbered from 0. |
| `-l, --level` | `4` | The level of the MCMC: one level up makes about three times fewer bins, and a run about three times faster. |
| `-n, --cycles` | `10` with `refine`, else `100` | The number of MCMC cycles: each visits every bin. Ignored without `mcmc`. |
| `-r, --resolutions` | `1000` | The resolutions of the `.mcool` file, in bp, comma-separated, each a multiple of a smaller one. |
| `-c, --coverage-std` | `1.0` | The coverage filter of the MCMC: fragments in contact with fewer others than the mean minus this many standard deviations are merged into the next fragment of their contig. |
| `-N, --neighborhood` | `5` | How many partner bins the MCMC draws for each bin, at random, weighted by their contacts with it. |
| `-C, --circular` | off | The genome is circular (MCMC). |
| `--save-matrix/--no-save-matrix` | on | Draw the contact map after each MCMC cycle. |
| `--balance/--no-balance` | on | Balance each resolution of the `.mcool` file (ICE). |
| `--balance-args` | | Options of `cooler.balance_cooler`, as JSON: `'{"max_iters": 2000, "mad_max": 10}'`. |
| `-s, --min-scaffold-size` | `0` | `instagraal-polish`: discard the scaffolds of this many bins or fewer. |
| `-L, --min-scaffold-length` | `0` | `instagraal-polish`: discard the scaffolds shorter than this, in bp. |
| `-j, --junction` | `NNNNNN` | The sequence between contigs, in the scaffolds of `instagraal-polish` and `instagraal-refine`; its length goes to `instagraal-post`. |
| `--criterion` | `blocks` | `instagraal-polish`: how runs of bins are reoriented, `blocks` or `sequences`. |
| `--cool-name` | the name of `PAIRS` | The base name of the cool and mcool files of `instagraal-pre` and `instagraal-post`. |
| `--simple` | off | MCMC: visit only the bins at the ends of the scaffolds; no traces are written. |
| `--save-pickle` | off | MCMC: write the state of the run to `graal.pkl`. |
| `--quiet` | off | MCMC: print only warnings and errors. |
| `--debug` | off | MCMC: print everything; overrides `--quiet`. |
| `--dry-run` | off | Print the commands without running them (no GPU needed). |
| `-b, --bomb/--no-bomb` | | Deprecated: the `legacy` pipeline, with or without bombing (`--no-bomb` is `--mode mcmc,polish`). Not with `--mode`. |
| `--polish/--no-polish` | | Deprecated: the `legacy` pipeline, with or without polishing (`--no-polish` is `--mode bomb,mcmc`). Not with `--mode`. |

## Steps

### `instagraal-pre` { #instagraal-pre }

Cut the contigs into restriction fragments, and count the read pairs between
them. **Guide:** [Preparing the contacts](guides/pre.md).

```console
instagraal-pre [OPTIONS] FASTA PAIRS
```

- **Inputs**: `FASTA`, the contigs; `PAIRS`, the pairs aligned on them. The
  columns `chr1`, `pos1`, `chr2` and `pos2` of the `#columns` line are read,
  or else columns 2 to 5; pairs on other sequences are left out.
- **Outputs**: `fragments_list.txt`, `info_contigs.txt` and
  `abs_fragments_contacts_weighted.txt` (the input folder of `instagraal`),
  `<name>.cool` (what `instagraal-refine` reads) and `<name>_hic_map.png`
  (see [File formats](formats.md#the-contacts-of-the-fragments)). It first
  prints the statistics of the contigs.

| Option | Default | Description |
|---|---|---|
| `-e, --enzyme` | required | The restriction enzymes of the library, comma-separated. |
| `-o, --output-dir` | `.` | The directory to write to. |
| `--cool-name` | the name of `PAIRS` | The base name of the cool file and its picture: `yeast.pairs.gz` gives `yeast`. |

### `instagraal` { #instagraal }

Scaffold the bins of the contigs with the MCMC, on an NVIDIA GPU. Needs the
`gpu` extra. **Guide:** [Scaffolding on a GPU](guides/scaffolding.md).

```console
instagraal [OPTIONS] HIC_FOLDER REFERENCE.FA
```

- **Inputs**: `HIC_FOLDER`, the folder of `instagraal-pre`; `REFERENCE.FA`,
  the contigs.
- **Outputs**: in `--output-dir`, `pyramids/` and
  `<HIC_FOLDER name>/test_mcmc_<level>/`, with `genome.fasta`,
  `info_frags.txt` and the traces of the run, written after each cycle (see
  [Outputs](guides/scaffolding.md#outputs)); in the current directory, the log
  and `sparsity_plot.pdf`, `sparsity_hist.pdf`.

| Option | Default | Description |
|---|---|---|
| `-o, --output-dir` | `out` | The directory to write to. |
| `-l, --level` | `4` | The level of the pyramid to scaffold at: bins of about 3<sup>level</sup> fragments. |
| `-n, --cycles` | `100` | The number of cycles: each visits every bin. |
| `-c, --coverage-std` | `1.0` | The coverage filter: fragments in contact with fewer others than the mean minus this many standard deviations are merged into the next fragment of their contig. Applied when the pyramid is built. |
| `-N, --neighborhood` | `5` | How many partner bins each visit draws, at random, weighted by their contacts with the bin. |
| `--device` | `0` | The GPU to run on, numbered from 0. |
| `-C, --circular` | off | The genome is circular. |
| `-b, --bomb` | off | Start from the bins, each its own scaffold, rather than from the contigs. |
| `--pyramid-only` | off | Build the pyramid, and stop. |
| `--save-pickle` | off | Write the state of the run to `graal.pkl`, in the current directory. |
| `--save-matrix` | off | Draw the contact map after each cycle (`matrix_cycle_<n>.png`). |
| `--simple` | off | Visit only the bins at the ends of the scaffolds, and keep the parameters of the model. Writes `genome.fasta` and `info_frags.txt` only: no traces, and no `--save-matrix` pictures. |
| `--quiet` | off | Print only warnings and errors. |
| `--debug` | off | Print everything; overrides `--quiet`. |

### `instagraal-polish` { #instagraal-polish }

Correct the artefacts of an MCMC layout, and write its scaffolds. **Guide:**
[Polishing the scaffolds](guides/polishing.md).

```console
instagraal-polish [OPTIONS]
```

- **Modes**: `polishing` (the default) runs `rearrange`, `inversion2`,
  `reincorporation` and `fasta`; the others run one step (see
  [The modes](guides/polishing.md#the-modes)).
- **Outputs**: in `--output-dir`, `new_info_frags.txt`; `polished_genome.fa`
  (modes `polishing` and `fasta`); `contig_composition.png` (all modes but
  `fasta`). It prints how many scaffolds the filters keep.

| Option | Default | Description |
|---|---|---|
| `-i, --input` | required | The layout to polish: `info_frags.txt` of `instagraal`. |
| `-f, --fasta` | | The contigs, as given to `instagraal-pre`. Required by the modes `polishing`, `fasta` and `reincorporation`. |
| `-o, --output-dir` | `out` | The directory to write to, created if missing. |
| `-m, --mode` | `polishing` | `polishing`, `rearrange`, `inversion2`, `inversion`, `singleton`, `reincorporation` or `fasta`. |
| `-c, --criterion` | `colinear` or `blocks` | Mode `inversion`: `cis`, `colinear` or `contiguous`. Modes `inversion2` and `polishing`: `blocks` or `sequences`. |
| `-s, --min-scaffold-size` | `0` | Discard the scaffolds of this many bins or fewer. With `polishing`, their DNA is put back next to its neighbours on the contigs. |
| `-l, --min-scaffold-length` | `0` | Discard the scaffolds shorter than this, in bp. |
| `-j, --junction` | `NNNNNN` | The sequence between bins of different contigs (`''` for none). Give its length to `instagraal-post --junction`. |

### `instagraal-refine` { #instagraal-refine }

Join whole contigs from the contacts next to their ends, place the short ones,
recentre circular chromosomes, and score the joins of a layout. Needs no GPU.
**Guide:** [Refining the scaffolds](guides/refine.md).

```console
instagraal-refine [OPTIONS]
```

- **Inputs**: the cool file of `instagraal-pre` and the contigs; optionally, a
  layout to check.
- **Outputs**: in `--output-dir`, `refined_genome.fa`,
  `refined_info_frags.txt`, `junctions.tsv` and `refine_report.txt`, which it
  also prints (see [Outputs](guides/refine.md#outputs)).

| Option | Default | Description |
|---|---|---|
| `-i, --info-frags` | | A layout to check: `new_info_frags.txt` of `instagraal-polish`, or `info_frags.txt` of `instagraal`. Its joins are scored; it decides whether `--extend auto` runs. |
| `-c, --cool` | required | The cool file of `instagraal-pre`. |
| `-f, --fasta` | required | The contigs, as given to `instagraal-pre`. |
| `-o, --output-dir` | `refined` | The directory to write to, created if missing. |
| `-b, --binsize` | `5000` | The bin size of the contig contacts, in bp. |
| `-w, --window` | `50000` | The length next to each contig end whose contacts are compared, in bp. At least `--binsize`. |
| `-l, --min-contig-length` | `15000` | Contigs shorter than this are not joined to each other, but placed (`--place-short`). |
| `--min-score` | `10.0` | The minimum log-likelihood ratio of a join (junction against background). |
| `--min-margin` | `2.0` | The minimum lead of a join over each end's next best candidate. |
| `--place-short/--no-place-short` | on | Place each short contig, whole, where its contacts fit best; otherwise, make it a scaffold of its own. |
| `--short-margin` | `10.0` | The minimum lead of a short contig's best place over its next best. |
| `--extend` | `auto` | The second pass, over whole scaffolds and 200 kb windows: `auto` runs it if refine confirms at least 30% of the joins of `--info-frags`; `on` always; `off` never. |
| `--recenter/--no-recenter` | on | Cut each circular scaffold of at least 500 kb at its replication origin, found from its GC skew. |
| `--origin-first/--terminus-first` | origin first | Start recentred scaffolds at the origin, or at the terminus. |
| `-j, --junction` | none | The sequence between contigs in `refined_genome.fa`, such as `NNNNNN`. |
| `--keep-unresolved` | off | Also keep the joins of `--info-frags` whose ends found no confident partner, if they reach `--min-score`. Needs `--info-frags`. |
| `--qc-only` | off | Only score the joins of `--info-frags` (`junctions.tsv`, `refine_report.txt`). Needs `--info-frags`. |

### `instagraal-post` { #instagraal-post }

Lift the read pairs over to the scaffolds, and write their contact maps.
**Guide:** [Contact maps of the assembly](guides/post.md).

```console
instagraal-post [OPTIONS] PAIRS NEW_INFO_FRAGS
```

- **Inputs**: `PAIRS`, the pairs aligned on the contigs, as given to
  `instagraal-pre`; `NEW_INFO_FRAGS`, the layout of the scaffolds
  (`final_info_frags.txt`, `refined_info_frags.txt` or `new_info_frags.txt`).
- **Outputs**: in `--output-dir`, `<name>_scaffolds_binned.mcool`,
  `<name>_scaffolds.cool`, `<name>_contigs.cool` (if `PAIRS` has `#chromsize`
  lines), `<name>_lifted.pairs.gz`, and pictures (see
  [What it writes](guides/post.md#what-it-writes)).

| Option | Default | Description |
|---|---|---|
| `-r, --resolutions` | required | The resolutions of the `.mcool` file, in bp, comma-separated. The smallest is the base, which the pairs are binned at; each other must be a multiple of a smaller one. |
| `-o, --output-dir` | `.` | The directory to write to, created if missing. |
| `--cool-name` | the name of `PAIRS` | The base name of the output files. |
| `--junction` | `6` | The length of the junction between contigs in the scaffolds: 6 for `NNNNNN`, 0 for none. |
| `--balance/--no-balance` | on | Balance each resolution (ICE). |
| `--balance-args` | | Options of `cooler.balance_cooler`, as JSON: `'{"max_iters": 2000, "mad_max": 10}'`. |
| `--fasta` | | Not used yet: reserved for the GC content of the bins. |

### `instagraal-stats` { #instagraal-stats }

Print the statistics of one assembly, or of several side by side. **Guide:**
[Checking an assembly](guides/checking.md#statistics).

```console
instagraal-stats [OPTIONS] ASSEMBLY.FA [ASSEMBLY.FA ...]
```

The statistics: number of sequences, total, largest, shortest, mean and
median length, N50, L50, N90, L90 and GC content.

| Option | Default | Description |
|---|---|---|
| `-l, --labels` | the file names | The names of the columns, comma-separated, one per file. |

### `instagraal-test` { #instagraal-test }

Check a GPU installation: check the GPU, pycuda and `nvcc`, then run
`instagraal-pre`, `instagraal` (from the bins, `--save-matrix`),
`instagraal-polish`, `instagraal-post` and `instagraal-stats` on the yeast test
dataset, downloaded from Zenodo
([10.5281/zenodo.19813387](https://doi.org/10.5281/zenodo.19813387)). Needs the
`gpu` extra.

```console
instagraal-test [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `-w, --workdir` | a temporary directory | The directory of the data and outputs. A temporary one is deleted at the end, unless `--keep`. |
| `--keep` | off | Keep the working directory. |
| `--fasta` | | The FASTA file of the test dataset, if downloaded already. |
| `--pairs` | | The pairs file of the test dataset, if downloaded already. |
| `--device` | `0` | The GPU to run on. |
| `-l, --level` | `4` | The level of the MCMC. |
| `-n, --cycles` | `3` | The number of MCMC cycles. |
