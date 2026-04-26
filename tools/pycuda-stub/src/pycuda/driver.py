"""Stub pycuda.driver - raises RuntimeError on any actual use."""


STUB_ERROR = "pycuda stub: no CUDA available"


def init():
    raise RuntimeError(STUB_ERROR)


def get_version():
    return (0, 0, 0)


def mem_get_info():
    raise RuntimeError(STUB_ERROR)


class Device:
    @staticmethod
    def count():
        return 0

    def __init__(self, *args, **kwargs):
        raise RuntimeError(STUB_ERROR)
