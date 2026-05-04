"""Lesson 12: Loss functions and numerical stability."""

# Learning objectives:
# - Identify unstable probability-space loss calculations.
# - Use logits-space cross entropy and stable softmax patterns.
# - Apply label smoothing without changing batch or class axes.
# Mental model:
# - Stable losses avoid forming extreme probabilities too early. Prefer logits and log-sum-exp based primitives when possible.
# Common mistakes:
# - Taking `log` of probabilities that may be exactly zero.
# - Smoothing labels in a way that changes each row's probability mass.
# Recap:
# - Stable softmax remains finite for large logits.
# - Logits-space cross entropy returns finite scalar losses.
# - Smoothed label rows still sum to one.

import jax
import jax.numpy as jnp


def naive_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    exp_logits = jnp.exp(logits)
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)


def stable_softmax(logits: jnp.ndarray) -> jnp.ndarray:
    shift = logits - jnp.max(logits, axis=-1, keepdims=True)
    exp_shift = jnp.exp(shift)
    return exp_shift / jnp.sum(exp_shift, axis=-1, keepdims=True)


def nll_from_probs(probs: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    n = probs.shape[0]
    picked = probs[jnp.arange(n), labels]
    picked = jnp.clip(picked, 1e-9, 1.0)
    return -jnp.mean(jnp.log(picked))


def cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    n = logits.shape[0]
    return -jnp.mean(log_probs[jnp.arange(n), labels])


def binary_cross_entropy_with_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(
        jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def exercise_label_smoothing(one_hot: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    """Exercise: apply label smoothing to one-hot labels."""
    raise NotImplementedError("Implement this in solutions/lesson_12_solution.py")
