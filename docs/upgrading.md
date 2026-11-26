# Upgrading

## From 0.2.0 { #from-020 }

- **`instagraal-endtoend` runs another pipeline by default.** It ran the MCMC
  from the bins, 100 cycles, then polished; it now runs the MCMC from the
  contigs, 10 cycles, then refine (`--mode default`). `--mode legacy` runs the
  pipeline of 0.2.0. Whatever the mode, the final assembly is
  `final_genome.fa`, and its layout `final_info_frags.txt`, in the output
  directory (see [Running the pipeline](guides/pipeline.md#outputs)).

    | 0.2.0 | Now |
    |---|---|
    | `instagraal-endtoend …` | `instagraal-endtoend --mode legacy …` |
    | `instagraal-endtoend --no-bomb …` | `instagraal-endtoend --mode mcmc,polish …` |
    | `polished/polished_genome.fa` | `final_genome.fa` (a copy of the last step's) |

    `--bomb/--no-bomb` still works, with a warning.

- **pycuda is an extra.** `pip install instagraal` no longer installs pycuda:
  for the MCMC scaffolder, install `pip install "instagraal[gpu]"` (see
  [Installation](installation.md)). The Docker images have it.
- **`instagraal-polish --junction` defaults to `NNNNNN`.** It joined the bins
  of different contigs with nothing, while `instagraal-post --junction`
  assumed 6 bp: their defaults now agree. For the former FASTA files, run
  `instagraal-polish --junction ''` and `instagraal-post --junction 0`.
- **Python API**: `instagraal.instagraal.instagraal_class` no longer has
  `start_MCMC`, `start_MTM` and `setup_simu`, which could not run.
- **Docker**: the images of releases are tagged `<version>-cuda<cuda>` and
  `latest-cuda<cuda>`, and `latest` is `latest-cuda11.8.0` (see
  [GPUs, CUDA and Docker](guides/gpu.md#docker)).

The [Changelog](changelog.md) lists every change, among them the fixes to
polishing, which put back every base the scaffolder left out exactly once.

## From 0.1 { #from-01 }

instaGRAAL 0.2.0 rewrote the command line, dropped the OpenGL viewer, and
added the steps around the scaffolder.

| 0.1 | Now |
|---|---|
| `instagraal hic_folder contigs.fa out` | `instagraal hic_folder contigs.fa -o out` (`out` by default; 0.1 wrote into the installed package without it) |
| The scaffolding movie, in an OpenGL window | Gone; `--save-matrix` draws the contact map after each cycle |
| The matrices of `hicstuff -M graal` | `instagraal-pre contigs.fa pairs.gz -e ENZYME` makes them from the pairs; hicstuff's still work for `instagraal` |
| `instagraal-polish -m polishing -i info_frags.txt -f contigs.fa -o curated.fa` | `instagraal-polish -i info_frags.txt -f contigs.fa -o polished`: `polished/polished_genome.fa` and `polished/new_info_frags.txt` (0.1 wrote `new_info_frags.txt` in the current directory) |
| `instagraal-polish -m <mode> … -o new_info_frags.txt` | `-o` is a directory: `new_info_frags.txt` and `contig_composition.png` in it |
| `instagraal-polish -m plot` | `contig_composition.png`, written by every mode but `fasta` |
| `instagraal-polish -j` (none by default) | `NNNNNN` by default |
| pycuda built by hand, with OpenGL and Boost (`instagraal-setup`) | `pip install "instagraal[gpu]"` |
| Python 3.4 or later | Python 3.10 to 3.12 |

New since 0.1: `instagraal-pre`, `instagraal-refine`, `instagraal-post`,
`instagraal-endtoend`, `instagraal-stats` and `instagraal-test` (see the
[CLI reference](cli.md)).
