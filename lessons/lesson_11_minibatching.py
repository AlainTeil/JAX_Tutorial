"""Lesson 11: Mini-batching and simple data pipelines."""

# Learning objectives:
# - Create deterministic toy classification data.
# - Slice and shuffle mini-batches with explicit keys and indices.
# - Aggregate per-batch losses into epoch-level metrics.
# Mental model:
# - Mini-batching is bookkeeping around arrays. Make indices, shuffling, and aggregation explicit so training remains reproducible.
# Common mistakes:
# - Dropping or duplicating examples when the final batch is smaller.
# - Shuffling inputs and labels with different permutations.
# Recap:
# - Batch indices cover the dataset in predictable chunks.
# - Shuffled features and labels stay aligned.
# - Epoch mean loss is a scalar summary of batch losses.

import jax
import jax.numpy as jnp


def make_classification_data(n: int = 12) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.arange(n * 2, dtype=jnp.float32).reshape(n, 2) / 10.0
    y = (jnp.sum(x, axis=1) > 1.0).astype(jnp.float32)
    return x, y


def batch_indices(num_examples: int, batch_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + batch_size, num_examples))
        for start in range(0, num_examples, batch_size)
    ]


def get_batch(
    x: jnp.ndarray, y: jnp.ndarray, start: int, end: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    return x[start:end], y[start:end]


def shuffle_data(key: jax.Array, x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    perm = jax.random.permutation(key, x.shape[0])
    return x[perm], y[perm]


def logistic_forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    logits = x @ params["w"] + params["b"]
    return jax.nn.sigmoid(logits)


def batch_bce_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    p = logistic_forward(params, x).squeeze(-1)
    p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
    return -jnp.mean(y * jnp.log(p) + (1.0 - y) * jnp.log(1.0 - p))


def exercise_epoch_mean_loss(losses: jnp.ndarray) -> jnp.ndarray:
    """Exercise: aggregate batch losses into an epoch mean."""
    raise NotImplementedError("Implement this in solutions/lesson_11_solution.py")
