# Changelog

All notable changes to instaGRAAL. [Upgrading](upgrading.md) says what to
change in your commands.

## Unreleased

### Breaking changes
- **`instagraal-endtoend` runs `--mode default`**: the MCMC from the contigs,
  10 cycles, then `instagraal-refine`, instead of the MCMC from the bins, 100
  cycles, then polishing. `--mode legacy` runs the pipeline of 0.2.0.
  ([#53](https://github.com/koszullab/instaGRAAL/pull/53))
- **pycuda is the `gpu` extra.** `pip install instagraal` installs every step
  but the MCMC scaffolder, on any machine; `pip install "instagraal[gpu]"`
  adds pycuda. The Docker images install the extra.
  ([#53](https://github.com/koszullab/instaGRAAL/pull/53))
- **`instagraal-polish --junction` defaults to `NNNNNN`** (it was none), the
  junction that `instagraal-post --junction 6` and `instagraal-endtoend`
  assume: on its defaults, `polished_genome.fa` has `NNNNNN` between bins of
  different contigs. `--junction ''` and `instagraal-post --junction 0` give
  the former output. ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- Python API: `instagraal_class` no longer has `start_MCMC`, `start_MTM` and
  `setup_simu`, which called methods that no longer existed.
  ([#50](https://github.com/koszullab/instaGRAAL/pull/50))

### Added
- **`instagraal-refine`** joins whole contigs from the contacts next to their
  ends, places the contigs under 15 kb, closes circular chromosomes and cuts
  them at their replication origin, found from their GC skew, and joins whole
  scaffolds over 200 kb windows (`--extend`). It scores the joins of a layout
  (`junctions.tsv`, `refine_report.txt`, `--qc-only`), and needs no GPU.
  ([#53](https://github.com/koszullab/instaGRAAL/pull/53))
- **`instagraal-endtoend --mode`**: a preset, `default` (`mcmc,refine`), `cpu`
  (`refine,extend`, no GPU) or `legacy` (`bomb,mcmc,polish`), or steps
  separated by commas. The final assembly is copied to `final_genome.fa`, and
  its layout to `final_info_frags.txt`, whatever the mode. Above 5,000
  contigs, a run without `--mode` falls back to `legacy`.
  ([#53](https://github.com/koszullab/instaGRAAL/pull/53))
- The documentation, at <https://koszullab.github.io/instagraal/>, built with
  Zensical from `docs/`, in place of the Sphinx site on Read the Docs, with a
  showcase on 27 genomes binned from a mock-community metagenome.
- Releases follow `bump-my-version`, which keeps the version of the package,
  of the documentation and of the Docker image in step, and dates this
  changelog.

### Changed
- `instagraal-polish --mode fasta` applies the size filters, and also writes
  `new_info_frags.txt`. ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- Without pycuda, `instagraal`, `instagraal-test` and the modes of
  `instagraal-endtoend` with `mcmc` stop with a message naming the `gpu`
  extra (and `--mode cpu`, but for `instagraal-test`). ([#53](https://github.com/koszullab/instaGRAAL/pull/53))
- The Docker images of a release are tagged `<version>-cuda<cuda>` and
  `latest-cuda<cuda>`, and `latest` is `latest-cuda11.8.0`: the four CUDA
  builds of a release no longer overwrite one `latest` tag and one
  `v<version>` tag.

### Deprecated
- `instagraal-endtoend --bomb/--no-bomb`: use `--mode`. It sets the `legacy`
  pipeline, with a warning; so does `--polish/--no-polish`, added after 0.2.0.
  ([#53](https://github.com/koszullab/instaGRAAL/pull/53))

### Removed
- The Sphinx documentation (`doc/`), and `example/example.gif`, the movie of
  the OpenGL viewer that 0.2.0 removed.
- `scripts/instagraal-setup`, the installer of 0.1, which no longer worked
  (it read a `requirements.txt` that is gone): `pip install "instagraal[gpu]"`
  installs pycuda.

### Fixed
- Polishing puts back every base the scaffolder left out exactly once. Lost
  regions were off by one, a region next to two bins was inserted twice,
  regions were misplaced, and only the last left-over region of each contig
  was kept: with `--min-scaffold-size` or `--min-scaffold-length`, tens of kb
  were duplicated per genome. ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- The MCMC dropped the last line of the FASTA file: the end of the last
  contig, or all of it if on one line. ([#52](https://github.com/koszullab/instaGRAAL/pull/52))
- `genome.fasta` dropped the last base of scaffolds of 61k+1 bp.
  ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- `instagraal-polish` crashed with matplotlib 3.9 or later; ignored
  `--criterion` in the `polishing` mode; kept a newline in FASTA headers.
  ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- `instagraal --simple` crashed. ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- `instagraal-endtoend --junction ''` told `instagraal-post` the junction was
  6 bp long. ([#49](https://github.com/koszullab/instaGRAAL/pull/49))
- The genome helpers of the sampler (`modify_genome`, `explode_genome`,
  `genome_content`) failed on the `next` field of the GPU structure.
  ([#50](https://github.com/koszullab/instaGRAAL/pull/50))
- The help of `--resolutions` (`instagraal-post`, `instagraal-endtoend`):
  one value makes one resolution, not a series. The help of
  `instagraal-post` and `instagraal-test`, and the docstrings of the Python
  API, describe the outputs and inputs as they are.

## [0.2.0] - 2026-04-27

A major overhaul of instaGRAAL
([#47](https://github.com/koszullab/instaGRAAL/pull/47)).

### Breaking changes
- Python 3.10 to 3.12.
- The OpenGL viewer is gone, and Boost.Python with it: pycuda installs from
  PyPI.
- A new command line: `instagraal` takes its output directory with
  `-o/--output-dir`; `instagraal-polish` writes into a directory, polishes by
  default, and has no `plot` mode.

### Added
- `instagraal-pre`: the contact map of the restriction fragments, from the
  contigs and the pairs, for one or more enzymes.
- `instagraal-post`: the contact maps of the scaffolds, from the pairs lifted
  over to them.
- `instagraal-endtoend`: the pipeline in one command.
- `instagraal-stats`: the statistics of assemblies, side by side.
- `instagraal-test`: a test of a GPU installation on a yeast dataset.
- `instagraal-polish -l/--min-scaffold-length`.
- Docker images for four CUDA versions, at `ghcr.io/koszullab/instagraal`.

### Fixed
- The contig names of FASTA files with descriptions in their headers.

## 0.1.1 to 0.1.6 - 2018-11-22 to 2019-03-21

The releases of instaGRAAL 0.1 on PyPI: the scaffolder, `instagraal`, with
its OpenGL viewer, and `instagraal-polish`.
