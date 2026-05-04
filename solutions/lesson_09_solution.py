"""Solutions for Lesson 09 exercises."""

import jax.numpy as jnp


def mae_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = x @ params["w"] + params["b"]
    return jnp.mean(jnp.abs(pred - y))
