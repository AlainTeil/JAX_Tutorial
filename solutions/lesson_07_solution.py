"""Solutions for Lesson 07 exercises."""

import jax
import jax.numpy as jnp


def gaussian_stats(key: jax.Array, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    samples = jax.random.normal(key, shape=(n,))
    return jnp.mean(samples), jnp.std(samples)
