import jax.numpy as jnp

from lessons.lesson_01_why_jax import binary_cross_entropy
from lessons.lesson_07_randomness import dropout, make_key
from lessons.lesson_12_losses_stability import (
    binary_cross_entropy_with_logits,
    stable_softmax,
)


def test_binary_cross_entropy_numerically_stable_near_extremes():
    proba = jnp.array([1e-12, 1.0 - 1e-12], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0], dtype=jnp.float32)
    loss = binary_cross_entropy(proba, y)
    assert jnp.isfinite(loss)


def test_stable_softmax_large_logits():
    logits = jnp.array([[1000.0, 1001.0, 1002.0]], dtype=jnp.float32)
    probs = stable_softmax(logits)
    assert jnp.all(jnp.isfinite(probs))
    assert jnp.allclose(jnp.sum(probs, axis=1), jnp.array([1.0], dtype=jnp.float32))


def test_bce_with_logits_extreme_values_and_dropout_shape():
    logits = jnp.array([-100.0, 0.0, 100.0], dtype=jnp.float32)
    targets = jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32)
    loss = binary_cross_entropy_with_logits(logits, targets)
    assert jnp.isfinite(loss)

    x = jnp.ones((8,), dtype=jnp.float32)
    out = dropout(make_key(123), x, keep_prob=0.5)
    assert out.shape == x.shape
