"""Solutions for Lesson 26 exercises."""

from math import prod


def estimate_batch_bytes(batch_shape: tuple[int, ...], dtype_size: int) -> int:
    return int(prod(batch_shape) * dtype_size)
