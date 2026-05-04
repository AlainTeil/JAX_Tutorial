import jax.numpy as jnp

from lessons.lesson_18_sequence_scan import (
    cumulative_sum_scan,
    run_simple_rnn,
    sequence_mse,
)
from solutions.lesson_18_solution import reverse_scan_suffix_sum


def test_cumulative_sum_scan():
    xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    out = cumulative_sum_scan(xs)
    assert jnp.allclose(out, jnp.array([1.0, 3.0, 6.0], dtype=jnp.float32))


def test_run_simple_rnn_shape_and_loss():
    params = {
        "wx": jnp.ones((2, 3), dtype=jnp.float32),
        "wh": jnp.ones((3, 3), dtype=jnp.float32),
        "b": jnp.zeros((3,), dtype=jnp.float32),
    }
    h0 = jnp.zeros((3,), dtype=jnp.float32)
    xs = jnp.ones((4, 2), dtype=jnp.float32)
    hs = run_simple_rnn(params, h0, xs)
    assert hs.shape == (4, 3)
    assert sequence_mse(hs, hs) == 0.0


def test_solution_suffix_sum():
    xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    out = reverse_scan_suffix_sum(xs)
    assert jnp.allclose(out, jnp.array([6.0, 5.0, 3.0], dtype=jnp.float32))
