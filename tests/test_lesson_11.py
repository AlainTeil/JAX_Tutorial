import jax.numpy as jnp

from lessons.lesson_11_minibatching import (
    batch_bce_loss,
    batch_indices,
    get_batch,
    logistic_forward,
    make_classification_data,
)
from solutions.lesson_11_solution import epoch_mean_loss


def test_batch_index_partitioning():
    idx = batch_indices(num_examples=10, batch_size=4)
    assert idx == [(0, 4), (4, 8), (8, 10)]


def test_get_batch_and_loss_shape():
    x, y = make_classification_data(n=8)
    xb, yb = get_batch(x, y, 0, 4)
    assert xb.shape == (4, 2)
    assert yb.shape == (4,)

    params = {
        "w": jnp.zeros((2, 1), dtype=jnp.float32),
        "b": jnp.array([0.0], dtype=jnp.float32),
    }
    p = logistic_forward(params, xb)
    assert p.shape == (4, 1)

    loss = batch_bce_loss(params, xb, yb)
    assert loss > 0.0


def test_solution_epoch_mean_loss():
    losses = jnp.array([0.6, 0.4, 0.2], dtype=jnp.float32)
    assert jnp.isclose(epoch_mean_loss(losses), 0.4)
