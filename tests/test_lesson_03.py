import jax.numpy as jnp

from lessons.lesson_03_gradients import (
    mse_grads,
    scalar_grad,
    second_derivative,
    softmax_jacobian,
)
from solutions.lesson_03_solution import mae_grads


def test_scalar_gradients_match_closed_form():
    x = 2.0
    # d/dx (x^3 + 2x) = 3x^2 + 2 = 14 at x=2.
    assert jnp.isclose(scalar_grad(x), 14.0)
    # d2/dx2 (x^3 + 2x) = 6x = 12 at x=2.
    assert jnp.isclose(second_derivative(x), 12.0)


def test_mse_grad_shapes():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32)
    params = {"w": jnp.array(0.0, dtype=jnp.float32), "b": jnp.array(0.0, dtype=jnp.float32)}

    loss, grads = mse_grads(params, x, y)
    assert loss > 0.0
    assert grads["w"].shape == ()
    assert grads["b"].shape == ()


def test_softmax_jacobian_rows_sum_to_zero():
    logits = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    jac = softmax_jacobian(logits)
    row_sums = jnp.sum(jac, axis=1)
    assert jnp.allclose(row_sums, jnp.zeros_like(row_sums), atol=1e-5)


def test_solution_mae_grad_runs():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = jnp.array([2.0, 4.0, 6.0], dtype=jnp.float32)
    params = {"w": jnp.array(1.0, dtype=jnp.float32), "b": jnp.array(0.0, dtype=jnp.float32)}

    loss, grads = mae_grads(params, x, y)
    assert loss >= 0.0
    assert grads["w"].shape == ()
    assert grads["b"].shape == ()
