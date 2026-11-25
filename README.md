# instaGRAAL

![ ](example/example.gif "instaGRAAL demo")

[![PyPI version](https://badge.fury.io/py/instagraal.svg)](https://badge.fury.io/py/instagraal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/instagraal.svg)
[![Read the docs](https://readthedocs.org/projects/instagraal/badge)](https://instagraal.readthedocs.io)
[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs13059--020--02041--z-blue)](https://doi.org/10.1186/s13059-020-02041-z)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)

Large genome reassembly based on Hi-C data (continuation and partial rewrite of [GRAAL](https://github.com/koszullab/GRAAL)).
It relies on `pycuda` for GPU-accelerated MCMC scaffolding and `cooler` for multi-resolution contact map generation.
Requires **Python 3.10-3.12** and the **CUDA toolkit** (to install `pycuda`). The default pipeline runs its MCMC on an **NVIDIA GPU**; `instagraal-endtoend --mode cpu` needs none.

## Installation

`instaGRAAL` is available on PyPI:

```sh
pip install instagraal
```

Pre-built Docker images are also available with `python3.12` and various `CUDA` versions:

```sh
docker run --gpus all -v /path/to/data:/work ghcr.io/koszullab/instagraal:latest-cuda11.8.0 \
  instagraal-endtoend assembly.fa reads.pairs -e DpnII -o out
```

**Note: Choosing the right CUDA version**

The CUDA version in the container must be compatible with the NVIDIA driver
installed on your host machine. Check your driver's maximum supported CUDA version by running:

```sh
nvidia-smi
```

Then select a container with that CUDA version or lower. For example, if your
driver supports up to CUDA 11.8, use `ghcr.io/koszullab/instagraal:latest-cuda11.8.0`.

When in doubt, `latest-cuda11.8.0` is the safest choice — it is
compatible with most drivers from the last few years and supports
a wide range of GPU generations.

## Usage

### Simplest approach

`instagraal-endtoend` runs the full pipeline — pre-processing, scaffolding, refinement, and contact-map generation — in a single command:

```sh
instagraal-endtoend assembly.fa reads.pairs --enzyme DpnII -o out
```

> `reads.pairs` must be a [4DN pairs file](https://github.com/4dn-dcic/pairix/blob/master/pairs_format_specification.md) produced by a Hi-C aligner such as [hicstuff](https://github.com/koszullab/hicstuff).

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `-e / --enzyme` | _(required)_ | Restriction enzyme(s), e.g. `DpnII` or `DpnII,HinfI` |
| `-o / --output-dir` | `out` | Root directory for all outputs |
| `-m / --mode` | `default` | Steps to run (see below) |
| `-l / --level` | `4` | Resolution level (higher = faster but coarser) |
| `-n / --cycles` | `10` with refine, else `100` | MCMC iterations (more = better convergence) |
| `--device` | `0` | CUDA device index |
| `-C / --circular` | off | Circular genome |

Run `instagraal-endtoend -h` for the full option list.

`--mode` chooses the steps run between pre-processing and contact-map generation: a preset, or steps separated by commas (`mcmc`, `bomb`, `polish`, `refine`, `extend` or `noextend`), in any order.

| Mode | Steps | GPU | What it does |
|------|-------|-----|--------------|
| `default` | `mcmc,refine` | yes | MCMC scaffolding from the contigs, then `instagraal-refine` (step 3b); refine's second, long-range pass runs only if refine confirms the MCMC layout |
| `cpu` | `refine,extend` | no | `instagraal-refine` alone, its second pass forced: the most contiguous assemblies, though on clearly incomplete bins that pass can add misjoins (`--mode refine` leaves it out) |
| `legacy` | `bomb,mcmc,polish` | yes | The pipeline before refine: MCMC on the genome exploded into bins, then `instagraal-polish` |

Other combinations work too, e.g. `--mode mcmc,polish` (no bombing) or `--mode mcmc` (the MCMC scaffolds as they are). Refine needs memory in proportion to the square of the number of contigs: above 5,000 contigs, the default mode falls back to `legacy`. Whatever the mode, the final assembly is also copied to `out/final_genome.fa`, and its layout to `out/final_info_frags.txt`. The former `--no-bomb` and `--no-polish` still work, with their former meaning on the legacy pipeline, but are deprecated.

### Step-by-step

Each stage can also be run independently if you need finer control.

**1. Pre-process** — convert a FASTA + pairs file into an instaGRAAL-compatible Hi-C folder:

```sh
instagraal-pre assembly.fa reads.pairs --enzyme DpnII -o hic_folder
```

Output: `hic_folder/` containing `fragments_list.txt`, `info_contigs.txt`, `abs_fragments_contacts_weighted.txt`, and a `.cool` file.

**2. Scaffold** — run the MCMC scaffolder:

```sh
instagraal hic_folder assembly.fa -o mcmc_out
```

Output: `mcmc_out/test_mcmc_4/` containing `genome.fasta` and `info_frags.txt`.

**3. Polish** — correct scaffolding artefacts (strongly recommended):

```sh
instagraal-polish -i mcmc_out/test_mcmc_4/info_frags.txt -f assembly.fa -o polished
```

Output: `polished/polished_genome.fa` and `polished/new_info_frags.txt`. Bins from different input contigs are joined with an `NNNNNN` junction (`--junction` sets another sequence, `--junction ''` none).

**3b. Refine** (optional, recommended for bacteria) — check every contig junction against the Hi-C contacts next to the two contig ends, and re-join whole contigs where the evidence is clear:

```sh
instagraal-refine -i polished/new_info_frags.txt -c hic_folder/reads.cool -f assembly.fa -o refined
```

In many bacteria the two replication arms of a chromosome are aligned, which adds an X (anti-diagonal) to the contact map. The scaffolder's model has no term for it and tends to fold such chromosomes, interleaving their arms: the map then looks deceptively smooth, and borders appear after polishing. `instagraal-refine` ignores long-range contacts, so it is not misled by the X. Contigs shorter than `--min-contig-length` (15 kb) are too short to be joined reliably: each is then placed, whole, where its contacts clearly fit best in the new scaffolds (`--short-margin`, or `--no-place-short` to leave them as scaffolds of their own). It writes `refined/refined_info_frags.txt` and `refined/refined_genome.fa` (use them in steps 4 and 5), plus `junctions.tsv` and `refine_report.txt`, a reference-free QC of the input layout (contradicted or unsupported joins, and GC skew of large scaffolds: a whole circular chromosome changes skew sign at most twice: at the origin and terminus). Use `--qc-only` to only check a layout.

Scaffolds of at least 500 kb whose two ends join (whole circular chromosomes) are recentered: cut at the replication origin found from their GC skew, so that they read like reference genomes, with a positive skew from the start to the terminus near the middle and a negative one after it (`--terminus-first` to cut at the terminus instead, which puts the origin in the middle; `--no-recenter` to leave them as they are). Skew cannot tell the two strands apart: read from the origin, both show this pattern.

Refine only joins contigs whose junction is clear from local contacts, so where the evidence is weak (gaps, low coverage) a chromosome can come out in several scaffolds. With the scaffolder's layout as input, a second pass (`--extend auto`, the default) joins whole scaffolds from contacts over 200 kb windows. Long-range contacts are exactly what aligned replication arms distort, so this pass only runs when refine confirms at least 30% of the input layout's joins, which a scaffolder folded by the X does not reach (`--extend on`/`off` to force it).

`-i` is optional: without it, `instagraal-pre` then `instagraal-refine -c hic_folder/reads.cool -f assembly.fa -o refined` gives an assembly without a GPU, joined from local evidence only (more fragmented where junctions are weak, as the second pass needs the scaffolder's layout to decide whether long-range contacts can be trusted).

**4. Post-process** — lift over contacts to the new assembly and build a multi-resolution `.mcool`:

```sh
instagraal-post reads.pairs polished/new_info_frags.txt -r 1000,5000,10000 -o post
```

`instagraal-post` rebuilds the polished coordinates from `new_info_frags.txt`, so its `--junction` must be the length of the junction used in step 3. The default (`6`) matches the default `NNNNNN`; after `instagraal-polish --junction ''`, pass `--junction 0`.

**5. Compare assemblies** — print statistics side-by-side:

```sh
instagraal-stats assembly.fa polished/polished_genome.fa \
  --labels "Input,Polished"
```

## Output

After scaffolding (`instagraal`), the output directory contains:

- `genome.fasta` — scaffolded assembly (scaffolds ordered by size in fragments)
- `info_frags.txt` — per-scaffold fragment table (`chromosome, id, orientation, start, end`); used as input for polishing and post-processing

After polishing (`instagraal-polish`):

- `polished_genome.fa` — final assembly after artefact correction
- `new_info_frags.txt` — updated fragment table for the polished coordinates

After refining (`instagraal-refine`):

- `refined_genome.fa` / `refined_info_frags.txt` — whole input contigs, re-joined where the junction evidence is clear (a recentered circular scaffold starts and ends with the two parts of the contig it is cut in)
- `junctions.tsv` / `refine_report.txt` — score and status of every input and refined join, and GC skew of large scaffolds

> Note: Scaffolding pyramids are cached in HDF5 format. If a run was interrupted and pyramids appear corrupted, delete the `pyramids/` folder and re-run.

## Contributing

```sh
git clone https://github.com/koszullab/instagraal.git
cd instagraal
uv sync --python 3.12          # creates .venv and installs all dependencies
uv run pytest                  # run the test suite
```

Format and lint before opening a PR:

```sh
uv run ruff format .
uv run ruff check .
```

Please open a pull request against `master` with a clear description of the change.

## Troubleshooting

### CUDA not found during installation

`pycuda` requires the **full CUDA Toolkit** (headers + compiler), not just the driver. Verify with:

```sh
nvcc --version
```

On an HPC cluster, load the module first:

```sh
module load cuda/12.9
```

On Ubuntu/Debian, install from [NVIDIA's official repository](https://developer.nvidia.com/cuda-downloads), then add to your environment:

```sh
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

Some Python dependencies also need system headers:

```sh
sudo apt-get install hdf5-tools libjpeg-dev zlib1g-dev
```

### CUDA library not found at runtime

```
ImportError: libcurand.so.X: cannot open shared object file
```

The CUDA libraries are not on `LD_LIBRARY_PATH`. Set it as shown above, or load the CUDA module on your cluster.

### Poor scaffolding quality

- Check the Hi-C mapping rate. Low rates often indicate poor contig completeness (run BUSCO) or missing `--iterative`/`--cutsite` flag in hicstuff.
- Verify that _trans_ contacts exist between contigs before scaffolding.
- Try switching aligner (bwa ↔ bowtie2) in hicstuff.
- For genomes > 500 Mb, use `--level 5`; for > 3 Gb, use `--level 6`.

### Slow scaffolding

Increase `--level` to trade resolution for speed (each step roughly triples the speed).

### KeyError on contig names

Spaces or special characters in contig names will cause errors. Rename contigs to alphanumeric identifiers before running.

## References

- [instaGRAAL: chromosome-level quality scaffolding of genomes using a proximity ligation-based scaffolder](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-020-02041-z) — Baudry et al., *Genome Biology*, 2020
- [High-quality genome assembly using chromosomal contact data](https://www.ncbi.nlm.nih.gov/pubmed/25517223) — Marie-Nelly et al., *Nature Communications*, 2014
- [A probabilistic approach for genome assembly from high-throughput chromosome conformation capture data](https://www.theses.fr/2013PA066714) — Marie-Nelly, PhD thesis, 2013

**Use cases:**
- [Proximity ligation scaffolding and comparison of two *Trichoderma reesei* strains](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5469131/) — Jourdier et al., *Biotechnology for Biofuels*, 2017
- [Scaffolding bacterial genomes and probing host-virus interactions in gut microbiome by proximity ligation](https://www.ncbi.nlm.nih.gov/pubmed/28232956) — Marbouty et al., *Science Advances*, 2017
