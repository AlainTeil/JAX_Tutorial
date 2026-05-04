"""Solutions for Lesson 04 exercises."""

import jax
import jax.numpy as jnp


def _relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(x, 0.0)


jitted_relu = jax.jit(_relu)
