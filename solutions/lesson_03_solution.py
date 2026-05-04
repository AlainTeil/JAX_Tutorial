"""Solutions for Lesson 03 exercises."""

import jax
import jax.numpy as jnp


def mae_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = x * params["w"] + params["b"]
    return jnp.mean(jnp.abs(preds - y))


def mae_grads(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray):
    return jax.value_and_grad(mae_loss)(params, x, y)
