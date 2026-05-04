"""Lesson 18: Sequence models with `lax.scan`."""

# Learning objectives:
# - Represent recurrent state updates as scan bodies.
# - Run a simple RNN over a sequence with explicit carry state.
# - Compute sequence losses with shape checks.
# Mental model:
# - A recurrent model is a scan: carry the hidden state, emit outputs, and keep sequence axes explicit.
# Common mistakes:
# - Mixing batch and time axes without naming the convention.
# - Closing over mutable hidden state instead of carrying it through scan.
# Recap:
# - Scan cumulative sums match simple examples.
# - RNN outputs keep one output per time step.
# - Reverse scan results align with suffix-sum intuition.

import jax.numpy as jnp
from jax import lax


def cumulative_sum_scan(xs: jnp.ndarray) -> jnp.ndarray:
    def step(carry, x):
        carry = carry + x
        return carry, carry

    _, ys = lax.scan(step, 0.0, xs)
    return ys


def simple_rnn_step(
    params: dict[str, jnp.ndarray], h: jnp.ndarray, x_t: jnp.ndarray
) -> jnp.ndarray:
    return jnp.tanh(x_t @ params["wx"] + h @ params["wh"] + params["b"])


def run_simple_rnn(params: dict[str, jnp.ndarray], h0: jnp.ndarray, xs: jnp.ndarray) -> jnp.ndarray:
    def step(h, x_t):
        h_next = simple_rnn_step(params, h, x_t)
        return h_next, h_next

    _, hs = lax.scan(step, h0, xs)
    return hs


def sequence_mse(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((pred - target) ** 2)


def exercise_reverse_scan(xs: jnp.ndarray) -> jnp.ndarray:
    """Exercise: compute suffix sums using scan over reversed input."""
    raise NotImplementedError("Implement this in solutions/lesson_18_solution.py")
