"""Solutions for Lesson 01 exercises."""

import jax
import jax.numpy as jnp


def accuracy_from_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
    return jnp.mean((preds == y).astype(jnp.float32))
