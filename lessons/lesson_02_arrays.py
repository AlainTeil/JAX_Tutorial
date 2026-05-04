"""Lesson 02: JAX arrays and immutable updates."""

# Learning objectives:
# - Create and inspect JAX arrays with predictable shapes and dtypes.
# - Use slicing, broadcasting, indexed updates, and matrix multiplication idiomatically.
# - Normalize rows without mutating the input array.
# Mental model:
# - JAX arrays are immutable values. Operations return new arrays, which is what makes transformations and device placement predictable.
# Common mistakes:
# - Expecting `x[index] = value` style mutation to work inside transformed code.
# - Letting broadcasting hide an unintended rank or axis mismatch.
# Recap:
# - Immutable updates leave the original input unchanged.
# - Broadcasted operations have the expected matrix shape.
# - Normalized row sums are close to one.

import jax.numpy as jnp


def create_core_arrays() -> dict[str, jnp.ndarray]:
    return {
        "vector": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32),
        "matrix": jnp.arange(1, 10, dtype=jnp.float32).reshape(3, 3),
        "zeros": jnp.zeros((2, 2), dtype=jnp.float32),
        "ones": jnp.ones((2, 2), dtype=jnp.float32),
    }


def slicing_example(matrix: jnp.ndarray) -> jnp.ndarray:
    return matrix[:2, 1:]


def broadcasting_example(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return a + b


def immutable_set(x: jnp.ndarray, index: int, value: float) -> jnp.ndarray:
    """JAX arrays are immutable; `.at[]` returns a new updated array."""
    return x.at[index].set(value)


def immutable_add_column(matrix: jnp.ndarray, col: int, value: float) -> jnp.ndarray:
    return matrix.at[:, col].add(value)


def matmul_with_einsum(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("ik,kj->ij", a, b)


def exercise_row_normalize(matrix: jnp.ndarray) -> jnp.ndarray:
    """Exercise: normalize each row to sum to one."""
    raise NotImplementedError("Implement this in solutions/lesson_02_solution.py")
