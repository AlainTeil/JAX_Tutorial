"""Solutions for Lesson 16 exercises."""

import jax.numpy as jnp


def classification_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean((preds == labels).astype(jnp.float32))
