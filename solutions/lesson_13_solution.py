"""Solutions for Lesson 13 exercises."""

import jax
import jax.numpy as jnp


def directional_derivative(x: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
    def f(t: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(t**2)

    grad = jax.grad(f)(x)
    return jnp.dot(grad, direction)
