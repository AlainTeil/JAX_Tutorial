"""Lesson 27: Data input pipelines."""

# Learning objectives:
# - Split deterministic records into train and validation partitions.
# - Create mini-batch slices without losing feature-label alignment.
# - Standardize evaluation data using statistics from the training split only.
# Mental model:
# - Input pipelines are part of the model contract: shapes, ordering, padding, and normalization choices decide what the compiled step receives.
# Common mistakes:
# - Computing normalization statistics on validation or test examples.
# - Letting a short final batch break a compiled step that expects fixed shapes.
# Recap:
# - Train and validation splits preserve matching feature and label counts.
# - Mini-batch iteration covers examples predictably with or without dropping the last batch.
# - Padded batches have a leading axis divisible by the target batch size.

import jax.numpy as jnp


def make_toy_records(n: int = 10) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.stack([jnp.linspace(-1.0, 1.0, n), jnp.linspace(1.0, -1.0, n)], axis=1).astype(
        jnp.float32
    )
    y = (x[:, 0] > x[:, 1]).astype(jnp.int32)
    return x, y


def train_val_split(
    x: jnp.ndarray, y: jnp.ndarray, val_fraction: float = 0.2
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    split_at = int(x.shape[0] * (1.0 - val_fraction))
    return (x[:split_at], y[:split_at]), (x[split_at:], y[split_at:])


def batch_slices(num_examples: int, batch_size: int, drop_last: bool = False) -> list[slice]:
    stops_at = num_examples if not drop_last else (num_examples // batch_size) * batch_size
    return [
        slice(start, min(start + batch_size, num_examples))
        for start in range(0, stops_at, batch_size)
    ]


def iterate_minibatches(
    x: jnp.ndarray, y: jnp.ndarray, batch_size: int, drop_last: bool = False
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    return [(x[idx], y[idx]) for idx in batch_slices(x.shape[0], batch_size, drop_last)]


def standardize_from_train(
    train_x: jnp.ndarray, eval_x: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = jnp.mean(train_x, axis=0, keepdims=True)
    std = jnp.maximum(jnp.std(train_x, axis=0, keepdims=True), 1e-6)
    return (train_x - mean) / std, (eval_x - mean) / std, mean, std


def exercise_pad_to_batch(x: jnp.ndarray, batch_size: int, pad_value: float = 0.0) -> jnp.ndarray:
    """Exercise: pad examples so the leading axis is divisible by batch size."""
    raise NotImplementedError("Implement this in solutions/lesson_27_solution.py")
