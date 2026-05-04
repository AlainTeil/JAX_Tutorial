import jax.numpy as jnp

from lessons.lesson_14_debugging_profiling import (
    finite_difference_grad,
    has_nans,
    jitted_polynomial_map,
    polynomial_jaxpr,
    polynomial_map,
    timed_run,
)
from solutions.lesson_14_solution import count_nans


def test_jitted_matches_eager():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    assert jnp.allclose(polynomial_map(x), jitted_polynomial_map(x))


def test_jaxpr_contains_mul_or_add():
    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    rep = str(polynomial_jaxpr(x))
    assert ("mul" in rep) or ("add" in rep)


def test_timed_run_and_nan_utilities():
    x = jnp.ones((1000,), dtype=jnp.float32)
    y, elapsed = timed_run(polynomial_map, x)
    assert y.shape == x.shape
    assert elapsed >= 0.0

    z = jnp.array([1.0, jnp.nan, jnp.nan], dtype=jnp.float32)
    assert has_nans(z)
    assert count_nans(z) == 2


def test_finite_difference_grad_close_to_true_grad():
    approx = finite_difference_grad(2.0)
    # d/dx x^3 at x=2 is 12
    assert abs(approx - 12.0) < 0.1
