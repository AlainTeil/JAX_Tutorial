"""Solutions for Lesson 02 exercises."""

import jax.numpy as jnp


def row_normalize(matrix: jnp.ndarray) -> jnp.ndarray:
    row_sums = jnp.sum(matrix, axis=1, keepdims=True)
    row_sums = jnp.where(row_sums == 0.0, 1.0, row_sums)
    return matrix / row_sums
