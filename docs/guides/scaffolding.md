# Scaffolding on a GPU

`instagraal` is the MCMC scaffolder: it orders and orients the bins of the
contigs to fit a model of the contacts, on an NVIDIA GPU (see
[Concepts](../concepts.md#the-model-and-the-mcmc)). It needs the `gpu` extra,
which installs pycuda (see [Installation](../installation.md#with-the-mcmc-scaffolder-gpu)).

```bash
instagraal pre contigs.fa --output-dir mcmc --level 4 --cycles 100
```

- `pre` is the folder of `instagraal-pre`, with `fragments_list.txt`,
  `info_contigs.txt` and `abs_fragments_contacts_weighted.txt` (see
  [Preparing the contacts](pre.md)).
- `contigs.fa` is the FASTA file of the same contigs.

Without pycuda, `instagraal` stops at once and says which extra to install;
`instagraal-endtoend --mode cpu` scaffolds without it.

## Options

| Option | Default | What it does |
|---|---|---|
| `-l, --level` | 4 | The level of the pyramid to scaffold at: bins of about 3<sup>level</sup> fragments. One level up makes about three times fewer bins, and a run about three times faster. |
| `-n, --cycles` | 100 | How many times each bin is visited. Too few and the layout does not converge; many more bring little. |
| `-N, --neighborhood` | 5 | How many of its most contacting bins each visit tries to bring a bin next to. |
| `-c, --coverage-std` | 1 | The coverage filter: fragments in contact with fewer others than the mean minus this many standard deviations are left out. |
| `-b, --bomb` | off | Start from the bins, each its own scaffold, rather than from the contigs. |
| `-C, --circular` | off | The genome is circular. |
| `--device` | 0 | The GPU to run on, numbered from 0, when there are several. |

The size of the genome sets the level: 4 suits genomes up to a few hundred
megabases; above 500 Mb, `--level 5` keeps the run short, and above 3 Gb,
`--level 6`. On the yeast test data, level 4 makes 727 bins of 16.6 kb on
average (see [Concepts](../concepts.md#levels-and-bins)).

Other options:

- `--simple` visits only the bins at the ends of the scaffolds, and keeps the
  parameters of the model as they are.
- `--pyramid-only` builds the pyramid and stops.
- `--save-matrix` draws the contact map after each cycle
  (`matrix_cycle_<n>.png`); `instagraal-endtoend` turns it on.
- `--save-pickle` writes the state of the run to `graal.pkl`, in the current
  directory, for development.
- `--quiet` prints only warnings and errors; `--debug` prints everything.

## Outputs

The output directory (`--output-dir`, `out` by default) gets the pyramid and
a folder named after the input folder and the level:

```text
mcmc/
├── pyramids/
│   ├── pyramid_1_no_thresh/     the fragments, before the filter
│   └── pyramid_9_thresh_auto/   after it: one folder per level
└── pre/test_mcmc_4/
    ├── genome.fasta             the scaffolds
    ├── info_frags.txt           their layout, bin by bin
    ├── list_likelihood.txt      the traces of the run
    ├── list_n_contigs.txt, …
    ├── save_simu_step_<n>.txt   the layout after each cycle
    └── matrix_cycle_<n>.png     with --save-matrix
```

- `genome.fasta` and `info_frags.txt` are written after each cycle: an
  interrupted run leaves the layout of its last cycle. The scaffolds are named
  `3C-assembly|contig_<n>`.
- The `list_*.txt` files trace the run, one value per move: the likelihood,
  the number of scaffolds and their mean length, the distance from the
  initial layout, and the parameters of the model. `list_mutations.txt` gives
  the moves.
- In the current directory, `instagraal` also writes its log,
  `instagraal-<date>.log`, and two plots of the coverage filter,
  `sparsity_plot.pdf` and `sparsity_hist.pdf`.

The scaffolds of the MCMC split some contigs and leave out the DNA of the
filtered fragments: polish or refine them (see
[Polishing the scaffolds](polishing.md) and
[Refining the scaffolds](refine.md)). [File formats](../formats.md#info_frags-files)
describes `info_frags.txt`.

## The pyramid

The pyramid is built once, then read again: a second run on the same output
directory skips building it. The coverage filter is applied as it is built,
so a later run keeps it whatever its own `--coverage-std`: to change the
filter, delete `pyramids/` or write to another directory. A run interrupted
while building the pyramid can leave a broken HDF5 file: delete `pyramids/`
and run again.

## Watching a run

The likelihood (`list_likelihood.txt`) should rise and level off, and the
number of scaffolds (`list_n_contigs.txt`) fall and level off. If they still
move at the last cycle, more cycles may help.

```python
import matplotlib.pyplot as plt
import numpy as np

likelihood = np.loadtxt("mcmc/pre/test_mcmc_4/list_likelihood.txt")
plt.plot(likelihood)
plt.xlabel("Move")
plt.ylabel("Log-likelihood")
plt.savefig("likelihood.png")
```
