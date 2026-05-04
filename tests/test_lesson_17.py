import jax.numpy as jnp

from lessons.lesson_17_regularization_eval import (
    accuracy_from_logits,
    linear_logits,
    precision_recall_from_logits,
    total_loss,
)
from solutions.lesson_17_solution import f1_score


def test_total_loss_with_weight_decay_is_higher():
    params = {
        "w": jnp.array([[1.0], [1.0]], dtype=jnp.float32),
        "b": jnp.array([0.0], dtype=jnp.float32),
    }
    x = jnp.array([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0], dtype=jnp.float32)
    no_reg = total_loss(params, x, y, weight_decay=0.0)
    with_reg = total_loss(params, x, y, weight_decay=0.1)
    assert with_reg > no_reg


def test_metrics_and_solution_f1():
    logits = jnp.array([-2.0, 2.0, 1.0, -1.0], dtype=jnp.float32)
    y = jnp.array([0.0, 1.0, 1.0, 0.0], dtype=jnp.float32)
    acc = accuracy_from_logits(logits, y)
    p, r = precision_recall_from_logits(logits, y)
    f1 = f1_score(p, r)
    assert jnp.isclose(acc, 1.0)
    assert jnp.isclose(p, 1.0)
    assert jnp.isclose(r, 1.0)
    assert jnp.isclose(f1, 1.0)


def test_linear_logits_shape():
    params = {
        "w": jnp.array([[1.0], [2.0]], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    x = jnp.array([[1.0, 1.0]], dtype=jnp.float32)
    out = linear_logits(params, x)
    assert out.shape == (1,)
