import jax.numpy as jnp

from lessons.lesson_21_tensor_sharding import (
    place_replicated,
    place_sharded_batch,
    sharded_linear,
)
from solutions.lesson_21_solution import verify_named_sharding


def test_place_replicated_and_verify_solution():
    x = jnp.array([[1.0, 2.0]], dtype=jnp.float32)
    x_rep = place_replicated(x)
    assert jnp.allclose(x_rep, x)
    assert verify_named_sharding(x_rep)


def test_place_sharded_batch_and_linear():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    w = jnp.array([[5.0], [6.0]], dtype=jnp.float32)
    x_sh = place_sharded_batch(x)
    out = sharded_linear(x_sh, w)
    assert jnp.allclose(out, x @ w)
