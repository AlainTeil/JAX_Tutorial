"""Lesson 07: Random numbers and PRNG keys in JAX."""

# Learning objectives:
# - Create, split, and pass explicit PRNG keys.
# - Sample deterministic random arrays for tests and demos.
# - Apply dropout without hidden global RNG state.
# Mental model:
# - A JAX key is an explicit input, not a mutable global generator. Splitting keys creates independent streams you can pass through pure functions.
# Common mistakes:
# - Reusing the same key and mistaking repeated samples for randomness.
# - Hiding key creation inside model code where tests cannot control it.
# Recap:
# - Splitting one key yields reproducible but distinct samples.
# - Dropout preserves shape and scales kept activations.
# - Gaussian summary statistics are near the expected mean and scale.

import jax
import jax.numpy as jnp


def make_key(seed: int) -> jax.Array:
    return jax.random.key(seed)


def split_three(key: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    return tuple(jax.random.split(key, 3))


def sample_normal(key: jax.Array, shape: tuple[int, ...]) -> jnp.ndarray:
    return jax.random.normal(key, shape=shape)


def sample_bernoulli(key: jax.Array, p: float, shape: tuple[int, ...]) -> jnp.ndarray:
    return jax.random.bernoulli(key, p=p, shape=shape).astype(jnp.float32)


def dropout(key: jax.Array, x: jnp.ndarray, keep_prob: float) -> jnp.ndarray:
    mask = sample_bernoulli(key, p=keep_prob, shape=x.shape)
    return (x * mask) / keep_prob


def random_batch(key: jax.Array, batch_size: int, dim: int) -> jnp.ndarray:
    return sample_normal(key, shape=(batch_size, dim))


def exercise_gaussian_stats(key: jax.Array, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Exercise: return mean and std of n standard-normal samples."""
    raise NotImplementedError("Implement this in solutions/lesson_07_solution.py")
