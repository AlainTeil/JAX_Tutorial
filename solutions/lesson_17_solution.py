"""Solutions for Lesson 17 exercises."""

import jax.numpy as jnp


def f1_score(precision: jnp.ndarray, recall: jnp.ndarray) -> jnp.ndarray:
    return 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-8)
