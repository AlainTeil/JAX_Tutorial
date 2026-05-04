"""Lesson 14: Profiling and debugging JAX programs."""

# Learning objectives:
# - Inspect traced programs with JAXPR.
# - Time eager and jitted functions without trusting the first compile call.
# - Detect NaNs and validate gradients with finite differences.
# Mental model:
# - Debugging JAX often means separating compile-time tracing from runtime execution. Inspect the staged program and synchronize before timing.
# Common mistakes:
# - Timing a jitted function without accounting for asynchronous dispatch or compilation.
# - Looking for Python side effects instead of inspecting arrays and JAXPR.
# Recap:
# - The JAXPR contains primitive operations rather than Python source lines.
# - NaN checks identify exactly the invalid entries.
# - Finite-difference gradients are close to autodiff for smooth scalar examples.

import time

import jax
import jax.numpy as jnp


def polynomial_map(x: jnp.ndarray) -> jnp.ndarray:
    return 3.0 * x**2 + 2.0 * x + 1.0


def jitted_polynomial_map(x: jnp.ndarray) -> jnp.ndarray:
    return jax.jit(polynomial_map)(x)


def polynomial_jaxpr(x: jnp.ndarray):
    return jax.make_jaxpr(polynomial_map)(x)


def timed_run(fn, x: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    start = time.perf_counter()
    y = fn(x)
    y.block_until_ready()
    elapsed = time.perf_counter() - start
    return y, elapsed


def has_nans(x: jnp.ndarray) -> bool:
    return bool(jnp.any(jnp.isnan(x)))


def finite_difference_grad(x: float, eps: float = 1e-4) -> float:
    def f(t: float) -> float:
        return t**3

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def exercise_count_nans(x: jnp.ndarray) -> int:
    """Exercise: count NaN entries in an array."""
    raise NotImplementedError("Implement this in solutions/lesson_14_solution.py")
