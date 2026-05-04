"""Solutions for Lesson 18 exercises."""

import jax.numpy as jnp
from jax import lax


def reverse_scan_suffix_sum(xs: jnp.ndarray) -> jnp.ndarray:
    rev = xs[::-1]

    def step(carry, x):
        carry = carry + x
        return carry, carry

    _, ys_rev = lax.scan(step, 0.0, rev)
    return ys_rev[::-1]
