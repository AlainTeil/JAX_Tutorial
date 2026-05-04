"""Solutions for Lesson 24 exercises."""

import jax.numpy as jnp


def confusion_matrix(preds: jnp.ndarray, labels: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    return jnp.zeros((num_classes, num_classes), dtype=jnp.int32).at[labels, preds].add(1)
