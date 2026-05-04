"""Solutions for Lesson 11 exercises."""

import jax.numpy as jnp


def epoch_mean_loss(losses: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(losses)
