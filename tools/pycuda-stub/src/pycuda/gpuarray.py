"""Stub pycuda.gpuarray."""


class GPUArray:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("pycuda stub: no CUDA available")


def to_gpu(*args, **kwargs):
    raise RuntimeError("pycuda stub: no CUDA available")


def zeros(*args, **kwargs):
    raise RuntimeError("pycuda stub: no CUDA available")
