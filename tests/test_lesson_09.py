import jax.numpy as jnp

from lessons.lesson_09_linear_models import (
    init_linear_params,
    linear_forward,
    make_regression_data,
    mse_loss,
    train_linear_model,
)
from solutions.lesson_09_solution import mae_loss


def test_regression_data_shapes():
    x, y = make_regression_data()
    assert x.shape == (4, 1)
    assert y.shape == (4, 1)


def test_linear_training_reduces_loss():
    x, y = make_regression_data()
    params = init_linear_params()
    initial = mse_loss(params, x, y)
    trained_params, losses = train_linear_model(num_steps=80, lr=0.1)
    final = mse_loss(trained_params, x, y)
    assert float(final) < float(initial)
    assert float(losses[-1]) < float(losses[0])


def test_solution_mae_loss_nonnegative():
    x, y = make_regression_data()
    params = {"w": jnp.array([[2.0]], dtype=jnp.float32), "b": jnp.array([1.0], dtype=jnp.float32)}
    pred = linear_forward(params, x)
    assert jnp.allclose(pred, y)
    assert jnp.isclose(mae_loss(params, x, y), 0.0)
