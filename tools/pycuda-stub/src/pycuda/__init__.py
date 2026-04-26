"""Stub pycuda package for CPU-only CI environments.

This package satisfies the ``pycuda`` pip dependency without requiring a
CUDA toolkit or GPU.  All symbols raise RuntimeError if actually invoked;
they are never called during non-GPU tests.
"""

VERSION = (2024, 1, 0)
VERSION_TEXT = "2024.1 (stub - no CUDA)"
