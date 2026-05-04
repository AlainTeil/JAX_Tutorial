import jax.numpy as jnp

from lessons.lesson_13_higher_order import (
    first_derivative,
    hessian_scalar_at,
    jacobian_vector_fn,
    second_derivative,
)
from solutions.lesson_13_solution import directional_derivative


def test_scalar_derivatives_match_closed_form():
    x = 2.0
    # f(x)=x^4+3x^2 => f'(x)=4x^3+6x, f''(x)=12x^2+6
    assert jnp.isclose(first_derivative(x), 44.0)
    assert jnp.isclose(second_derivative(x), 54.0)


def test_jacobian_shape_and_values():
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    jac = jacobian_vector_fn(x)
    assert jac.shape == (2, 2)
    expected = jnp.array([[2.0, 1.0], [jnp.cos(1.0), 12.0]], dtype=jnp.float32)
    assert jnp.allclose(jac, expected, atol=1e-6)


def test_hessian_and_directional_derivative_solution():
    assert jnp.isclose(hessian_scalar_at(1.0), 18.0)

    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    d = jnp.array([3.0, 4.0], dtype=jnp.float32)
    # grad(sum(x^2)) at x is 2x => [2,4], dot d => 22
    assert jnp.isclose(directional_derivative(x, d), 22.0)
