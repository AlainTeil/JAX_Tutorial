import jax.numpy as jnp

from lessons.lesson_02_arrays import (
    broadcasting_example,
    create_core_arrays,
    immutable_add_column,
    immutable_set,
    matmul_with_einsum,
    slicing_example,
)
from solutions.lesson_02_solution import row_normalize


def test_core_array_shapes():
    arrays = create_core_arrays()
    assert arrays["vector"].shape == (3,)
    assert arrays["matrix"].shape == (3, 3)


def test_slicing_and_broadcasting():
    matrix = jnp.arange(1, 10, dtype=jnp.float32).reshape(3, 3)
    sliced = slicing_example(matrix)
    assert sliced.shape == (2, 2)

    a = jnp.array([[1.0], [2.0]], dtype=jnp.float32)
    b = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)
    out = broadcasting_example(a, b)
    assert out.shape == (2, 3)
    assert jnp.allclose(out[0], jnp.array([11.0, 21.0, 31.0], dtype=jnp.float32))


def test_immutable_updates_and_einsum():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    y = immutable_set(x, 1, 9.0)
    assert jnp.allclose(x, jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32))
    assert jnp.allclose(y, jnp.array([1.0, 9.0, 3.0], dtype=jnp.float32))

    m = jnp.ones((2, 3), dtype=jnp.float32)
    m2 = immutable_add_column(m, col=1, value=2.0)
    assert jnp.allclose(m2[:, 1], jnp.array([3.0, 3.0], dtype=jnp.float32))

    a = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    b = jnp.array([[5.0], [6.0]], dtype=jnp.float32)
    assert jnp.allclose(matmul_with_einsum(a, b), a @ b)


def test_solution_row_normalize():
    mat = jnp.array([[1.0, 1.0], [1.0, 3.0]], dtype=jnp.float32)
    out = row_normalize(mat)
    assert jnp.allclose(jnp.sum(out, axis=1), jnp.ones((2,), dtype=jnp.float32))
