"""Solutions for Lesson 27 exercises."""

import jax.numpy as jnp


def pad_to_batch(x: jnp.ndarray, batch_size: int, pad_value: float = 0.0) -> jnp.ndarray:
    remainder = x.shape[0] % batch_size
    if remainder == 0:
        return x
    pad_rows = batch_size - remainder
    pad_width = [(0, pad_rows), *[(0, 0) for _ in x.shape[1:]]]
    return jnp.pad(x, pad_width, constant_values=pad_value)
