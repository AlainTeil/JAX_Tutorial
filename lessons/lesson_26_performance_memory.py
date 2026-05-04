"""Lesson 26: Performance and memory measurement."""

# Learning objectives:
# - Measure compiled work only after synchronizing asynchronous JAX execution.
# - Separate compilation cost from repeated execution cost in small benchmarks.
# - Estimate memory use from pytree leaves and batch shapes.
# Mental model:
# - JAX dispatch is asynchronous, so timing without `block_until_ready` often measures enqueue time rather than compute time.
# Common mistakes:
# - Timing the first jitted call and treating compilation as steady-state performance.
# - Ignoring activation and batch memory when choosing model or batch size.
# Recap:
# - Jitted and eager matrix-stack outputs have the same shape.
# - Blocking runs produce a numeric result and elapsed time.
# - Estimated batch bytes equal the product of shape dimensions times dtype size.

import time
from functools import partial

import jax
import jax.numpy as jnp


def make_benchmark_inputs(n: int = 8) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.arange(n * n, dtype=jnp.float32).reshape(n, n) / float(n)
    w = jnp.eye(n, dtype=jnp.float32) * 0.5 + 0.1
    return x, w


def matmul_stack(x: jnp.ndarray, w: jnp.ndarray, depth: int = 3) -> jnp.ndarray:
    out = x
    for _ in range(depth):
        out = jnp.tanh(out @ w)
    return out


@partial(jax.jit, static_argnames=("depth",))
def jitted_matmul_stack(x: jnp.ndarray, w: jnp.ndarray, depth: int = 3) -> jnp.ndarray:
    return matmul_stack(x, w, depth=depth)


def blocking_sum(x: jnp.ndarray) -> jnp.ndarray:
    result = jnp.sum(x)
    result.block_until_ready()
    return result


def timed_blocking_run(fn, *args, **kwargs) -> tuple[jnp.ndarray, float]:
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    result.block_until_ready()
    return result, time.perf_counter() - start


def tree_nbytes(tree) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(leaf.size * leaf.dtype.itemsize for leaf in leaves))


def exercise_estimate_batch_bytes(batch_shape: tuple[int, ...], dtype_size: int) -> int:
    """Exercise: estimate the number of bytes in a dense batch tensor."""
    raise NotImplementedError("Implement this in solutions/lesson_26_solution.py")
