"""Lesson 06: Control flow with `lax.cond` and `lax.scan`."""

# Learning objectives:
# - Use `lax.cond` for data-dependent branches.
# - Use `lax.scan` for loop-carried state over sequences.
# - Recognize when Python control flow should become JAX control flow.
# Mental model:
# - Transformed JAX code needs control flow that can be represented in the traced program rather than decided by Python at trace time.
# Common mistakes:
# - Branching on a traced array with a Python `if`.
# - Using a Python loop for sequence state that should compose with `jit` or gradients.
# Recap:
# - Safe inverse avoids division blowups around zero.
# - Scan cumulative sums match the intuitive running total.
# - Weighted cumulative sums preserve sequence length.

import jax.numpy as jnp
from jax import lax


def safe_inverse(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Use `lax.cond` so branch logic is JIT-friendly."""
    return lax.cond(
        jnp.abs(x) > eps,
        lambda t: 1.0 / t,
        lambda _: 0.0,
        x,
    )


def cumulative_sum_scan(xs: jnp.ndarray) -> jnp.ndarray:
    def step(carry, x):
        new_carry = carry + x
        return new_carry, new_carry

    _, ys = lax.scan(step, 0.0, xs)
    return ys


def running_mean_scan(xs: jnp.ndarray) -> jnp.ndarray:
    def step(carry, x):
        total, count = carry
        total = total + x
        count = count + 1.0
        mean = total / count
        return (total, count), mean

    (_, _), means = lax.scan(step, (0.0, 0.0), xs)
    return means


def rnn_step(params: dict[str, jnp.ndarray], h: jnp.ndarray, x_t: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(x_t @ params["wx"] + h @ params["wh"] + params["b"])


def rnn_unroll(params: dict[str, jnp.ndarray], h0: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    def step(h, x_t):
        h_next = rnn_step(params, h, x_t)
        return h_next, h_next

    _, hs = lax.scan(step, h0, xs)
    return hs


def exercise_weighted_cumsum(xs: jnp.ndarray, alpha: float) -> jnp.ndarray:
    """Exercise: implement weighted cumulative sum using lax.scan."""
    raise NotImplementedError("Implement this in solutions/lesson_06_solution.py")
