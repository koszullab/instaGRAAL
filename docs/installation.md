# Installation

instaGRAAL needs Python 3.10 to 3.12. Every command installs from PyPI and
runs on any machine, except the MCMC scaffolder, which needs an NVIDIA GPU, the
CUDA toolkit and pycuda.

## With pip

```bash
pip install instagraal
```

This installs every step but the MCMC scaffolder: `instagraal-endtoend --mode cpu`,
`instagraal-pre`, `instagraal-refine`, `instagraal-polish`, `instagraal-post`
and `instagraal-stats`. Install it in its own environment, so that it does not
change the versions other tools rely on.

!!! tip "Check it worked"

    ```bash
    instagraal-endtoend --version
    ```

## With the MCMC scaffolder (GPU)

On a machine with an NVIDIA GPU and the
[CUDA toolkit](https://developer.nvidia.com/cuda-downloads), the `gpu` extra
adds [pycuda](https://pypi.org/project/pycuda/), which `instagraal`,
`instagraal-test` and the modes of `instagraal-endtoend` with `mcmc` need:

```bash
pip install "instagraal[gpu]"
```

pip builds pycuda against the toolkit: `nvcc --version` must work first. On a
cluster, load the toolkit's module before installing (`module load cuda`).
[GPUs, CUDA and Docker](guides/gpu.md) says more.

!!! tip "Check the GPU"

    `instagraal-test` downloads a small yeast dataset and runs the scaffolding
    pipeline on it, on the GPU:

    ```bash
    instagraal-test
    ```

## With Docker

Images with pycuda and CUDA are published to the GitHub Container Registry,
one per CUDA version (11.8.0, 12.4.1, 12.8.1 and 13.2.1):

```bash
docker pull ghcr.io/koszullab/instagraal:latest-cuda11.8.0
docker run --gpus all -v "$PWD":/work \
    ghcr.io/koszullab/instagraal:latest-cuda11.8.0 \
    instagraal-endtoend contigs.fa pairs.gz --enzyme DpnII --output-dir out
```

The CUDA version of the image must be at most the one your NVIDIA driver
supports, which `nvidia-smi` prints. `latest-cuda11.8.0` runs on most drivers
of recent years. See [GPUs, CUDA and Docker](guides/gpu.md#docker) for the
tags, and for Apptainer.

!!! note "Until the release after 0.2.0"

    The `latest-cuda<version>` tags are pushed from the release after 0.2.0
    on. Until then, use the images of the development version:
    `ghcr.io/koszullab/instagraal:master-cuda11.8.0`.

## Development version

The development version is managed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/koszullab/instaGRAAL.git
cd instaGRAAL
uv sync --extra test                # without a GPU
uv sync --extra test --extra gpu    # with the CUDA toolkit
uv run instagraal-endtoend --help
```

`uv run --only-group docs zensical serve` previews this documentation; see
[Contributing](contributing.md).

## Licence

instaGRAAL is distributed under the GNU General Public License v3 (GPLv3).
