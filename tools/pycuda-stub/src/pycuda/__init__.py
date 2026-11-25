"""Stub pycuda package for CPU-only CI environments.

instagraal installs without pycuda (it is in the ``gpu`` extra), but its MCMC
module imports pycuda.  This stub lets that module import without a CUDA
toolkit or GPU, so that the CPU CI can test how it dispatches.  All symbols
raise RuntimeError if actually invoked; they are never called during non-GPU
tests.
"""

VERSION = (2024, 1, 0)
VERSION_TEXT = "2024.1 (stub - no CUDA)"
