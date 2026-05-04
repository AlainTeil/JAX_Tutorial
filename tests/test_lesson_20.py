import jax.numpy as jnp

from lessons.lesson_20_pjit_partitioning import (
    inspect_array_sharding,
    make_mesh,
    pjit_matmul,
    replicate_sharding,
    shard_first_dim,
)
from solutions.lesson_20_solution import make_sharded_array


def test_mesh_and_shardings_construct():
    with make_mesh() as mesh:
        rep = replicate_sharding(mesh)
        sh = shard_first_dim(mesh)
        assert rep is not None
        assert sh is not None


def test_pjit_matmul_matches_eager():
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    w = jnp.array([[5.0], [6.0]], dtype=jnp.float32)
    out = pjit_matmul(x, w)
    assert jnp.allclose(out, x @ w)


def test_solution_make_sharded_array():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = make_sharded_array(x)
    assert jnp.allclose(y, x)
    assert inspect_array_sharding(y) is not None
