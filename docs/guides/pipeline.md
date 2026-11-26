# Running the pipeline

`instagraal-endtoend` runs the whole pipeline: it makes the contact map of
the contigs, scaffolds them, writes the contact maps of the scaffolds and
compares the two assemblies. Each step is also a command, which the run
prints as it goes.

```bash
instagraal-endtoend contigs.fa pairs.gz --enzyme DpnII,HinfI --output-dir out
```

## Inputs

| Input | What it is |
|---|---|
| `FASTA` | The contigs, plain or gzip-compressed. Their names must hold no spaces or special characters. |
| `PAIRS` | The Hi-C read pairs aligned on the contigs, in the [pairs format](../formats.md#pairs), plain or gzip-compressed. hicstuff writes them (see [Preparing the contacts](pre.md)). |
| `--enzyme` | The restriction enzymes of the library, comma-separated: `DpnII`, `DpnII,HinfI`… Any enzyme Biopython knows. |

## Modes { #modes }

`--mode` says which steps run between `instagraal-pre` and
`instagraal-post`: a preset, or steps separated by commas, in any order.

| Mode (steps) | GPU | What it does |
|---|---|---|
| `default` (`mcmc,refine`) | yes | MCMC scaffolding from the contigs (10 cycles), then refine. Refine's second, long-range pass runs only if refine confirms at least 30% of the MCMC's joins. |
| `cpu` (`refine,extend`) | no | Refine alone, its second pass forced: the most contiguous assemblies, though on incomplete genomes that pass can add misjoins. `--mode refine` leaves it out. |
| `legacy` (`bomb,mcmc,polish`) | yes | The pipeline of instaGRAAL 0.2.0: MCMC from the bins (100 cycles), then polish. |

The steps:

| Step | What it runs |
|---|---|
| `mcmc` | The MCMC scaffolder, `instagraal`, on the GPU: from the contigs, or from the bins with `bomb`. Then `instagraal-polish`, which polishes the scaffolds with `polish`, or writes them as they are without it. |
| `bomb` | Start the MCMC from the bins, each its own scaffold, rather than from the contigs. Needs `mcmc`. |
| `polish` | Polish the MCMC scaffolds (`instagraal-polish`). Needs `mcmc`. |
| `refine` | `instagraal-refine`: join whole contigs from the contacts next to their ends; with `mcmc`, it also checks the MCMC layout. |
| `extend`, `noextend` | Force refine's second pass on or off (by default, it runs only if refine confirms the MCMC layout; without `mcmc`, it does not run). Need `refine`. |

`--mode mcmc` gives the MCMC scaffolds as they are, and `--mode mcmc,polish`
polishes them without bombing. A run prints a note when a step changes
little: with `extend` or `noextend`, the MCMC does not change the assembly;
and refine gives the same assembly with or without `polish` up to its second
pass, which `auto` decides from the layout it is given, polished or not.

### Choosing a mode

On 27 bacterial genomes binned from the metagenome of a mock community
(benchmark of [koszullab/instaGRAAL#53](https://github.com/koszullab/instaGRAAL/pull/53),
10 MCMC cycles in each mode), compared with the reference genomes of 23 of
them:

| Mode | Wrong joins | Median NA50 | Circular chromosomes closed |
|---|---|---|---|
| `legacy` | 508 of 850 (60%) | 11% of the genome | – |
| `default` | 132 of 563 (23%) | 17% | 10 of 27 |
| `cpu` | 158 of 604 (26%) | 25% | 12 of 27 |

A wrong join links contigs of 15 kb or more that are not neighbours on the
reference. NA50 is the N50 of the blocks between breakpoints (where a scaffold
jumps more than 50 kb on the reference), as a share of the genome's length.
The legacy scaffolds are long but chimeric. `default` and `cpu` gave the same assembly
on 18 of the 27 genomes; `cpu` is more contiguous where `default` does not
trust the MCMC layout enough to run the second pass. The
[Showcase](../showcase.md) shows these assemblies, genome by genome.

### More than 5,000 contigs

Refine compares every contig end with every other: its memory grows with the
square of the number of contigs (some 6 GB at 5,000). Above 5,000 contigs, a
run without `--mode` falls back to `legacy`, with a warning, and a run whose
`--mode` includes `refine` is refused.

## Outputs { #outputs }

In the output directory (`--output-dir`, `out` by default), in the `default`
mode:

```text
out/
├── final_genome.fa         the scaffolds, whatever the mode
├── final_info_frags.txt    their layout
├── pre/                    instagraal-pre
│   ├── fragments_list.txt
│   ├── info_contigs.txt
│   ├── abs_fragments_contacts_weighted.txt
│   ├── <name>.cool
│   └── <name>_hic_map.png
├── mcmc/                   instagraal (mcmc)
│   ├── pyramids/           the contact maps of each level
│   └── pre/test_mcmc_4/    the MCMC layout and its traces
├── polished/               instagraal-polish
│   ├── polished_genome.fa
│   └── new_info_frags.txt
├── refined/                instagraal-refine (refine)
│   ├── refined_genome.fa
│   ├── refined_info_frags.txt
│   ├── junctions.tsv
│   └── refine_report.txt
└── post/                   instagraal-post
    ├── <name>_scaffolds_binned.mcool
    ├── <name>_scaffolds.cool
    ├── <name>_contigs.cool
    ├── <name>_lifted.pairs.gz
    └── <name>_*.png
```

`<name>` is `--cool-name`, or the name of the pairs file without `.pairs` and
`.gz`. `final_genome.fa` and `final_info_frags.txt` are copies of the last
step's: refine's in `default` and `cpu`, polish's in `legacy`. Without `mcmc`,
there is no `mcmc/` nor `polished/`; without `refine`, no `refined/`. The MCMC
also writes its log (`instagraal-<date>.log`) and two plots of the coverage
filter (`sparsity_*.pdf`) in the current directory. [File formats](../formats.md)
describes every file.

## Options

Each option goes to one step:

| Option | Step | Default |
|---|---|---|
| `-e, --enzyme`, `--cool-name` | `instagraal-pre` (and `--cool-name` to `instagraal-post`) | required; the pairs file's name |
| `-l, --level`, `-n, --cycles`, `-c, --coverage-std`, `-N, --neighborhood`, `--device`, `-C, --circular`, `--save-matrix/--no-save-matrix`, `--simple`, `--save-pickle`, `--quiet`, `--debug` | `instagraal` (MCMC) | 4; 10 with refine, else 100; 1; 5; 0; off; on; off… |
| `-s, --min-scaffold-size`, `-L, --min-scaffold-length`, `--criterion` | `instagraal-polish` | 0; 0; `blocks` |
| `-j, --junction` | `instagraal-polish`, `instagraal-refine`, and its length to `instagraal-post` | `NNNNNN` |
| `-r, --resolutions`, `--balance/--no-balance`, `--balance-args` | `instagraal-post` | `1000`; balance |

The [CLI reference](../cli.md#instagraal-endtoend) describes each. The
options of the steps that a mode does not run are ignored.

## Running the steps one by one

A run ends with the commands it ran. `--dry-run` prints them without running
anything, and needs no GPU. In the `default` mode, they are:

```bash
instagraal-pre contigs.fa pairs.gz --enzyme DpnII,HinfI \
    --output-dir out/pre
instagraal out/pre contigs.fa --output-dir out/mcmc --level 4 \
    --cycles 10 --device 0 --coverage-std 1.0 --neighborhood 5 \
    --save-matrix
instagraal-polish --input out/mcmc/pre/test_mcmc_4/info_frags.txt \
    --fasta contigs.fa --output-dir out/polished --min-scaffold-size 0 \
    --min-scaffold-length 0 --mode fasta --junction NNNNNN
instagraal-refine --info-frags out/polished/new_info_frags.txt \
    --cool out/pre/pairs.cool --fasta contigs.fa --output-dir out/refined \
    --junction NNNNNN --extend auto
instagraal-post pairs.gz out/refined/refined_info_frags.txt \
    --resolutions 1000 --output-dir out/post --junction 6
instagraal-stats contigs.fa out/refined/refined_genome.fa \
    --labels "Input assembly,Refined assembly"
```

The guides take each step in turn: [Preparing the contacts](pre.md),
[Scaffolding on a GPU](scaffolding.md), [Refining the scaffolds](refine.md),
[Polishing the scaffolds](polishing.md) and
[Contact maps of the assembly](post.md).

## Earlier options

`-b, --bomb/--no-bomb` set the pipeline of 0.2.0, and `--polish/--no-polish`
that of the development version after it. Both still work, with their former
meaning on the `legacy` pipeline (`--no-bomb` gives `--mode mcmc,polish`,
`--no-polish` `--mode bomb,mcmc`), with a warning, and cannot be combined
with `--mode`. See [Upgrading](../upgrading.md).
