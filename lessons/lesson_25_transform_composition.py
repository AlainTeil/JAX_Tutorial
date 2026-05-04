"""Lesson 25: Transform composition patterns."""

# Learning objectives:
# - Compose `vmap`, `grad`, and `jit` without changing the mathematical contract.
# - Compare per-example gradients with gradients of a reduced batch objective.
# - Use a compiled update as an integration point for earlier transform lessons.
# Mental model:
# - Transform composition works best when each function has a crisp contract: one example, one batch, one loss, or one update.
# Common mistakes:
# - Changing the reduction axis and accidentally changing what the gradient means.
# - Compiling a large opaque training step before checking the small eager pieces.
# Recap:
# - Per-example gradient leaves keep the batch dimension first.
# - The compiled update returns the same parameter tree keys as the input.
# - Clipped gradients have global norm no larger than the requested limit.

import jax
import jax.numpy as jnp


def make_regression_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.array([[-1.0, 0.5], [0.0, 1.0], [1.0, 1.5], [2.0, 2.0]], dtype=jnp.float32)
    y = jnp.array([-1.5, -0.25, 1.0, 2.25], dtype=jnp.float32)
    return x, y


def init_params() -> dict[str, jnp.ndarray]:
    return {"w": jnp.array([0.2, -0.1], dtype=jnp.float32), "b": jnp.array(0.0)}


def predict(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def per_example_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return (predict(params, x) - y) ** 2


def batch_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    losses = jax.vmap(per_example_loss, in_axes=(None, 0, 0))(params, x, y)
    return jnp.mean(losses)


def per_example_gradients(
    params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    return jax.vmap(jax.grad(per_example_loss), in_axes=(None, 0, 0))(params, x, y)


@jax.jit
def compiled_update(
    params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    loss, grads = jax.value_and_grad(batch_loss)(params, x, y)
    updated = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return updated, loss


def transform_order_summary() -> dict[str, str]:
    return {
        "vmap_grad": "per-example gradients; maps a scalar-loss gradient over examples",
        "grad_vmap": "gradient of a batched objective; reduces examples before differentiating",
        "jit_update": "compiles the full update once shapes and dtypes are known",
    }


def exercise_clip_gradients(
    grads: dict[str, jnp.ndarray], max_norm: float
) -> dict[str, jnp.ndarray]:
    """Exercise: clip a gradient pytree by global L2 norm."""
    raise NotImplementedError("Implement this in solutions/lesson_25_solution.py")
