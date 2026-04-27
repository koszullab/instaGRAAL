# instaGRAAL

![ ](example/example.gif "instaGRAAL demo")

[![PyPI version](https://badge.fury.io/py/instagraal.svg)](https://badge.fury.io/py/instagraal)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/instagraal.svg)
[![Read the docs](https://readthedocs.org/projects/instagraal/badge)](https://instagraal.readthedocs.io)
[![DOI](https://img.shields.io/badge/DOI-10.1186%2Fs13059--020--02041--z-blue)](https://doi.org/10.1186/s13059-020-02041-z)
[![License: GPLv3](https://img.shields.io/badge/License-GPL%203-0298c3.svg)](https://opensource.org/licenses/GPL-3.0)

Large genome reassembly based on Hi-C data (continuation and partial rewrite of [GRAAL](https://github.com/koszullab/GRAAL)).
It relies on `pycuda` for GPU-accelerated MCMC scaffolding and `cooler` for multi-resolution contact map generation.
Requires **Python 3.10-3.12** and an **NVIDIA GPU** with the **CUDA toolkit** installed.

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

`instagraal-endtoend` runs the full pipeline — pre-processing, scaffolding, polishing, and contact-map generation — in a single command:

```sh
instagraal-endtoend assembly.fa reads.pairs --enzyme DpnII -o out
```

> `reads.pairs` must be a [4DN pairs file](https://github.com/4dn-dcic/pairix/blob/master/pairs_format_specification.md) produced by a Hi-C aligner such as [hicstuff](https://github.com/koszullab/hicstuff).

Key options:

| Option | Default | Description |
|--------|---------|-------------|
| `-e / --enzyme` | _(required)_ | Restriction enzyme(s), e.g. `DpnII` or `DpnII,HinfI` |
| `-o / --output-dir` | `out` | Root directory for all outputs |
| `-l / --level` | `4` | Resolution level (higher = faster but coarser) |
| `-n / --cycles` | `100` | MCMC iterations (more = better convergence) |
| `--device` | `0` | CUDA device index |
| `-C / --circular` | off | Circular genome |

Run `instagraal-endtoend -h` for the full option list.

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

Output: `polished/polished_genome.fa` and `polished/new_info_frags.txt`.

**4. Post-process** — lift over contacts to the new assembly and build a multi-resolution `.mcool`:

```sh
instagraal-post reads.pairs polished/new_info_frags.txt -r 1000,5000,10000 -o post
```

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
