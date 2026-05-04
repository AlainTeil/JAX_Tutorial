import jax.numpy as jnp

from lessons.lesson_12_losses_stability import (
    binary_cross_entropy_with_logits,
    cross_entropy_with_logits,
    naive_softmax,
    nll_from_probs,
    stable_softmax,
)
from solutions.lesson_12_solution import label_smoothing


def test_softmax_outputs_sum_to_one():
    logits = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    p_naive = naive_softmax(logits)
    p_stable = stable_softmax(logits)
    assert jnp.allclose(jnp.sum(p_naive, axis=1), jnp.array([1.0], dtype=jnp.float32))
    assert jnp.allclose(jnp.sum(p_stable, axis=1), jnp.array([1.0], dtype=jnp.float32))


def test_cross_entropy_matches_nll_from_probs():
    logits = jnp.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=jnp.float32)
    labels = jnp.array([0, 1], dtype=jnp.int32)
    probs = stable_softmax(logits)
    nll = nll_from_probs(probs, labels)
    ce = cross_entropy_with_logits(logits, labels)
    assert jnp.allclose(nll, ce, atol=1e-6)


def test_binary_cross_entropy_with_logits_positive():
    logits = jnp.array([0.0, 2.0, -2.0], dtype=jnp.float32)
    targets = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    loss = binary_cross_entropy_with_logits(logits, targets)
    assert loss > 0.0


def test_solution_label_smoothing():
    one_hot = jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32)
    smoothed = label_smoothing(one_hot, epsilon=0.1)
    assert jnp.allclose(jnp.sum(smoothed, axis=1), jnp.array([1.0], dtype=jnp.float32))
    assert smoothed[0, 0] < 1.0
