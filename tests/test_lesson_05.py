import jax.numpy as jnp

from lessons.lesson_05_vmap import (
    batch_squared_l2_loop,
    batch_squared_l2_vmap,
    batched_linear,
    pairwise_dot_batched,
)
from solutions.lesson_05_solution import batch_cosine_similarity


def test_vmap_matches_loop_for_squared_l2():
    batch = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    assert jnp.allclose(batch_squared_l2_loop(batch), batch_squared_l2_vmap(batch))


def test_batched_linear_output_shape():
    params = {
        "w": jnp.array([[1.0], [2.0]], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    x_batch = jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float32)
    out = batched_linear(params, x_batch)
    assert out.shape == (2, 1)
    assert jnp.allclose(out[:, 0], jnp.array([3.5, 6.5], dtype=jnp.float32))


def test_pairwise_dot_batched():
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]], dtype=jnp.float32)
    out = pairwise_dot_batched(a, b)
    assert jnp.allclose(out, jnp.array([17.0, 53.0], dtype=jnp.float32))


def test_solution_batch_cosine_similarity():
    a = jnp.array([[1.0, 0.0], [1.0, 1.0]], dtype=jnp.float32)
    b = jnp.array([[1.0, 0.0], [1.0, -1.0]], dtype=jnp.float32)
    out = batch_cosine_similarity(a, b)
    assert jnp.allclose(out[0], 1.0)
    assert jnp.allclose(out[1], 0.0, atol=1e-6)
