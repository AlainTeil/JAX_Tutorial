import jax
import jax.numpy as jnp

from lessons.lesson_07_randomness import (
    dropout,
    make_key,
    random_batch,
    sample_normal,
    split_three,
)
from solutions.lesson_07_solution import gaussian_stats


def test_key_split_shapes_and_uniqueness():
    key = make_key(0)
    k1, k2, k3 = split_three(key)
    assert k1.shape == ()
    assert k2.shape == ()
    assert k3.shape == ()
    assert not jnp.array_equal(jax.random.key_data(k1), jax.random.key_data(k2))


def test_sampling_is_reproducible_for_same_key():
    key = make_key(123)
    x1 = sample_normal(key, shape=(3,))
    x2 = sample_normal(key, shape=(3,))
    assert jnp.allclose(x1, x2)


def test_dropout_preserves_shape():
    key = make_key(42)
    x = jnp.ones((4,), dtype=jnp.float32)
    out = dropout(key, x, keep_prob=0.5)
    assert out.shape == x.shape


def test_random_batch_and_solution_stats():
    key = make_key(7)
    batch = random_batch(key, batch_size=5, dim=2)
    assert batch.shape == (5, 2)

    k_stats = make_key(99)
    mean, std = gaussian_stats(k_stats, n=20000)
    assert jnp.abs(mean) < 0.05
    assert jnp.abs(std - 1.0) < 0.05
