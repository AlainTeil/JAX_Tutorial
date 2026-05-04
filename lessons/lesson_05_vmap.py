"""Lesson 05: Vectorization with `jax.vmap`."""

# Learning objectives:
# - Lift a single-example function to batched inputs with `jax.vmap`.
# - Control mapped axes for pairwise operations.
# - Compare loop-style batching with transformation-style batching.
# Mental model:
# - `vmap` adds a batch axis to a function contract. The inner function should still describe one example clearly.
# Common mistakes:
# - Mapping over the wrong axis and silently comparing unrelated examples.
# - Writing a Python loop first and forgetting the simpler single-example function.
# Recap:
# - Loop and `vmap` L2 computations agree numerically.
# - Batched linear output keeps the batch dimension first.
# - Cosine similarity returns one score per paired example.

import jax
import jax.numpy as jnp


def squared_l2(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x**2)


def batch_squared_l2_loop(batch: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([squared_l2(x) for x in batch], dtype=batch.dtype)


def batch_squared_l2_vmap(batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(squared_l2)(batch)


def linear(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def batched_linear(params: dict[str, jnp.ndarray], x_batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(linear, in_axes=(None, 0))(params, x_batch)


def pairwise_dot(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.dot(a, b)


def pairwise_dot_batched(a_batch: jnp.ndarray, b_batch: jnp.ndarray) -> jnp.ndarray:
    return jax.vmap(pairwise_dot, in_axes=(0, 0))(a_batch, b_batch)


def exercise_batch_cosine_similarity(a_batch: jnp.ndarray, b_batch: jnp.ndarray) -> jnp.ndarray:
    """Exercise: compute per-example cosine similarity with vmap."""
    raise NotImplementedError("Implement this in solutions/lesson_05_solution.py")
