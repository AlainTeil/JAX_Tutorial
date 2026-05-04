"""Solutions for Lesson 14 exercises."""

import jax.numpy as jnp


def count_nans(x: jnp.ndarray) -> int:
    return int(jnp.sum(jnp.isnan(x)))
