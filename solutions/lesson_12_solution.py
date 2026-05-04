"""Solutions for Lesson 12 exercises."""

import jax.numpy as jnp


def label_smoothing(one_hot: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    num_classes = one_hot.shape[-1]
    return (1.0 - epsilon) * one_hot + (epsilon / num_classes)
