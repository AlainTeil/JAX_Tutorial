"""Solutions for Lesson 30 exercises."""

import jax.numpy as jnp


def apply_threshold(probs: jnp.ndarray, threshold: float) -> jnp.ndarray:
    return (probs >= threshold).astype(jnp.int32)
