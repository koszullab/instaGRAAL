# GPUs, CUDA and Docker

Only the MCMC scaffolder, `instagraal`, runs on a GPU: so do
`instagraal-test` and the modes of `instagraal-endtoend` with `mcmc`. It needs
an NVIDIA GPU, the CUDA toolkit and pycuda. Every other command runs on any
machine, and `instagraal-endtoend --mode cpu` scaffolds without a GPU.

## The CUDA toolkit and pycuda

pycuda is built against the CUDA toolkit when it is installed: the toolkit's
compiler and headers must be there first, not only the NVIDIA driver.

```bash
nvcc --version                     # the toolkit
pip install "instagraal[gpu]"      # instaGRAAL and pycuda
```

- **On a cluster**, load the toolkit's module first: `module load cuda` (or
  the version your cluster provides).
- **On Ubuntu or Debian**, install the toolkit from
  [NVIDIA's repository](https://developer.nvidia.com/cuda-downloads), then put
  it on your paths, in your shell's start-up file:

    ```bash
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

- Some dependencies build against system libraries:
  `sudo apt-get install hdf5-tools libjpeg-dev zlib1g-dev`.

`instagraal-test` checks the GPU, pycuda and `nvcc`, then runs the scaffolding
pipeline on a small yeast dataset. instaGRAAL's own GPU tests run on CUDA 12.1
and 12.4, on Ampere and Ada GPUs. Blackwell GPUs (RTX 50 series) need CUDA
12.8 or later.

## Choosing the GPU

`--device` picks the GPU, numbered from 0 as `nvidia-smi` lists them, on
`instagraal`, `instagraal-endtoend` and `instagraal-test`. The scaffolder
uses one GPU. `CUDA_VISIBLE_DEVICES` restricts the GPUs a run sees.

## Docker { #docker }

The images at `ghcr.io/koszullab/instagraal` hold Ubuntu, Python 3.12, CUDA
and instaGRAAL with pycuda. Each is built for one CUDA version: 11.8.0 and
12.4.1 (Ubuntu 22.04), 12.8.1 and 13.2.1 (Ubuntu 24.04).

| Tag | What it is |
|---|---|
| `latest-cuda<cuda>` | The newest release, e.g. `latest-cuda11.8.0` |
| `latest` | The same as `latest-cuda11.8.0` |
| `<version>-cuda<cuda>` | A release, e.g. `<version>-cuda12.4.1` |
| `master-cuda<cuda>` | The development version, rebuilt at each push to `master` |
| `sha-<commit>-cuda<cuda>` | One commit of `master` |

!!! note "Until the release after 0.2.0"

    The release tags are pushed from the release after 0.2.0 on. Until then,
    use the `master-cuda<cuda>` images: the `latest` and `v0.2.0` tags hold
    0.2.0, built for one of the four CUDA versions, which the tag does not say.

The CUDA version of the image must be at most the one the driver of the
machine supports, which `nvidia-smi` prints at its top right (`CUDA Version`).
`cuda11.8.0` runs on most drivers of recent years; Blackwell GPUs need
`cuda12.8.1` or later. Docker needs the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
to give a container the GPU (`--gpus all`):

```bash
docker run --gpus all -v "$PWD":/work \
    ghcr.io/koszullab/instagraal:latest-cuda11.8.0 \
    instagraal-endtoend contigs.fa pairs.gz --enzyme DpnII,HinfI \
    --output-dir out
```

The image works in `/work`: mount the directory of your data there.

### Apptainer

On clusters, Apptainer (formerly Singularity) runs the same images, with
`--nv` for the GPU:

```bash
apptainer pull instagraal.sif \
    docker://ghcr.io/koszullab/instagraal:latest-cuda11.8.0
apptainer exec --nv instagraal.sif \
    instagraal-endtoend contigs.fa pairs.gz --enzyme DpnII,HinfI \
    --output-dir out
```

## Without a GPU

`pip install instagraal` installs every step but the MCMC scaffolder.
`instagraal-endtoend --mode cpu` scaffolds with `instagraal-refine` alone; see
[Running the pipeline](pipeline.md#modes) for how its assemblies compare.
