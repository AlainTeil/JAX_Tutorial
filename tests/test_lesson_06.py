import jax.numpy as jnp

from lessons.lesson_06_control_flow import (
    cumulative_sum_scan,
    rnn_unroll,
    running_mean_scan,
    safe_inverse,
)
from solutions.lesson_06_solution import weighted_cumsum


def test_safe_inverse_with_cond():
    assert jnp.isclose(safe_inverse(jnp.array(2.0)), 0.5)
    assert jnp.isclose(safe_inverse(jnp.array(0.0)), 0.0)


def test_cumulative_sum_and_running_mean():
    xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    csum = cumulative_sum_scan(xs)
    means = running_mean_scan(xs)
    assert jnp.allclose(csum, jnp.array([1.0, 3.0, 6.0], dtype=jnp.float32))
    assert jnp.allclose(means, jnp.array([1.0, 1.5, 2.0], dtype=jnp.float32))


def test_rnn_unroll_shape():
    params = {
        "wx": jnp.ones((2, 3), dtype=jnp.float32),
        "wh": jnp.ones((3, 3), dtype=jnp.float32),
        "b": jnp.zeros((3,), dtype=jnp.float32),
    }
    h0 = jnp.zeros((3,), dtype=jnp.float32)
    xs = jnp.ones((4, 2), dtype=jnp.float32)
    hs = rnn_unroll(params, h0, xs)
    assert hs.shape == (4, 3)


def test_solution_weighted_cumsum():
    xs = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    out = weighted_cumsum(xs, alpha=0.5)
    expected = jnp.array([1.0, 1.5, 1.75], dtype=jnp.float32)
    assert jnp.allclose(out, expected)
