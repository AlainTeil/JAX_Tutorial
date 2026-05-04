"""Solutions for Lesson 05 exercises."""

import jax
import jax.numpy as jnp


def _cosine(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    denom = (jnp.linalg.norm(a) * jnp.linalg.norm(b)) + 1e-8
    return jnp.dot(a, b) / denom


def batch_cosine_similarity(a_batch: jnp.ndarray, b_batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(_cosine, in_axes=(0, 0))(a_batch, b_batch)
