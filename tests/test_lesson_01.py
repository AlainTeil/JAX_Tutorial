import jax.numpy as jnp

from lessons.lesson_01_why_jax import (
    gd_step,
    init_logreg_params,
    loss_fn,
    make_toy_batch,
    predict_logits,
)
from solutions.lesson_01_solution import accuracy_from_logits


def test_logreg_training_reduces_loss():
    x, y = make_toy_batch()
    params = init_logreg_params(x.shape[1])

    initial_loss = loss_fn(params, x, y)
    for _ in range(30):
        params, _ = gd_step(params, x, y, lr=0.2)

    final_loss = loss_fn(params, x, y)
    assert float(final_loss) < float(initial_loss)


def test_solution_accuracy_from_logits():
    x, y = make_toy_batch()
    params = {"w": jnp.array([2.0, 2.0], dtype=jnp.float32), "b": jnp.array(-2.0)}
    logits = predict_logits(params, x)
    acc = accuracy_from_logits(logits, y)
    assert 0.5 <= float(acc) <= 1.0
