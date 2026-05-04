"""Solutions for Lesson 23 exercises."""

import jax.numpy as jnp


def is_loss_decreasing(losses: jnp.ndarray) -> bool:
    return bool(losses[-1] < losses[0])
