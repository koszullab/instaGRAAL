# Troubleshooting

## "The MCMC scaffolder needs pycuda and a CUDA GPU"

`instagraal`, `instagraal-test` and the modes of `instagraal-endtoend` with
`mcmc` need pycuda, which `pip install instagraal` does not install. On a
machine with an NVIDIA GPU and the CUDA toolkit, install the `gpu` extra:
`pip install "instagraal[gpu]"`. Without a GPU, run
`instagraal-endtoend --mode cpu`.

## CUDA not found during installation

`pip install "instagraal[gpu]"` builds pycuda, which needs the whole CUDA
toolkit (its compiler and headers), not only the NVIDIA driver. Check that
`nvcc --version` works; on a cluster, load the toolkit's module first
(`module load cuda`); otherwise, see
[GPUs, CUDA and Docker](guides/gpu.md#the-cuda-toolkit-and-pycuda). The
Docker images hold a working toolkit and pycuda.

## CUDA library not found at runtime

```text
ImportError: libcurand.so.X: cannot open shared object file
```

The CUDA libraries are not on `LD_LIBRARY_PATH`: add the toolkit's `lib64`
directory to it (see [GPUs, CUDA and Docker](guides/gpu.md#the-cuda-toolkit-and-pycuda)),
or load the CUDA module of your cluster. If the version it asks for is not the
one installed, reinstall pycuda against the toolkit you have:
`pip install --no-cache-dir --force-reinstall pycuda`.

## KeyError on contig names

Spaces and special characters in contig names break the steps that match
names between files. Rename the contigs with letters, digits and underscores
(`contig_001`…), in the FASTA file and before aligning the Hi-C reads, so
that the pairs use the same names.

## Poor scaffolds

- **Check the contacts.** A low share of aligned reads, or few contacts
  between contigs in `pre/<name>_hic_map.png`, give poor scaffolds. Align the
  reads with hicstuff's `-m cutsite` or `-m iterative`; check the completeness
  of the contigs (with BUSCO); try the other aligner (bwa or bowtie2).
- **Check the contigs.** Refine keeps contigs whole: it cannot fix a
  misassembled contig. Contigs from noisy long reads gain from being polished
  before scaffolding.
- **Check the junctions.** `refined/junctions.tsv` scores each join (see
  [Checking an assembly](guides/checking.md)).
- **Compare the modes.** `--mode cpu` often gives more contiguous scaffolds
  than `default`, which runs refine's second pass only when the MCMC layout
  agrees with refine (see [Running the pipeline](guides/pipeline.md#choosing-a-mode)).

## Slow scaffolding

Each level up makes about three times fewer bins, and a run about three times
faster: `--level 5` for genomes above 500 Mb, `--level 6` above 3 Gb. With
refine after it (the `default` mode), the MCMC runs 10 cycles; on its own,
100. `--mode cpu` runs no MCMC.

## A changed `--coverage-std` makes no difference

The coverage filter is applied when the pyramid is built, and a later run in
the same output directory reuses the pyramid. Delete `pyramids/` in the output
directory of `instagraal` (`mcmc/pyramids/` for `instagraal-endtoend`), or
write to another directory.

## A run stopped while building the pyramid

The pyramid's HDF5 file may be broken: delete `pyramids/` and run again.

## The contact maps drift from the FASTA file

`instagraal-post --junction` must be the length of the junction between
contigs in the FASTA file of the scaffolds: 6 for `NNNNNN`, the default of
`instagraal-polish`, and 0 for `instagraal-refine` run on its own, which adds
no junction by default (see
[Contact maps of the assembly](guides/post.md#the-junction)).
`instagraal-endtoend` passes it on.

## "Resolution N cannot be derived from the base resolutions"

Each resolution of `instagraal-post --resolutions` must be a multiple of a
smaller one: `1000,5000,25000` works, `1000,2500` does not.

## "more than instagraal-refine handles"

Refine's memory grows with the square of the number of contigs: above 5,000
contigs, `instagraal-endtoend` without `--mode` runs `legacy`, and refuses a
mode with `refine`. Run `--mode legacy`, or another mode without `refine`.

## Still stuck?

Open an issue on [GitHub](https://github.com/koszullab/instaGRAAL/issues) with
the command, its output and the version (`instagraal-endtoend --version`).
