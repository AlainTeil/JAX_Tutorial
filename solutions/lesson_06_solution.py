"""Solutions for Lesson 06 exercises."""

import jax.numpy as jnp
from jax import lax


def weighted_cumsum(xs: jnp.ndarray, alpha: float) -> jnp.ndarray:
    def step(carry, x):
        new_carry = alpha * carry + x
        return new_carry, new_carry

    _, ys = lax.scan(step, 0.0, xs)
    return ys
