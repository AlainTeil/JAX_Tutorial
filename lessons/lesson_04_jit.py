"""Lesson 04: Compilation with `jax.jit`."""

# Learning objectives:
# - Compile pure numerical functions with `jax.jit`.
# - Separate array inputs from static Python choices that affect tracing.
# - Run a jitted training step that returns updated parameters and loss.
# Mental model:
# - `jit` traces Python once for a shape/dtype/static-argument signature, then reuses compiled device code for matching calls.
# Common mistakes:
# - Changing Python control decisions inside a jitted function without marking them static.
# - Expecting print/debug side effects to behave like eager Python during compilation.
# Recap:
# - The jitted train step returns the same tree structure as the input parameters.
# - Static choices such as `power` are explicit at the call boundary.
# - The ReLU exercise output has no negative values.

import jax
import jax.numpy as jnp


def affine_forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


jitted_affine_forward = jax.jit(affine_forward)


def mse_from_affine(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = affine_forward(params, x)
    return jnp.mean((pred - y) ** 2)


def train_step(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float):
    loss, grads = jax.value_and_grad(mse_from_affine)(params, x, y)
    new_params = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"],
    }
    return new_params, loss


jitted_train_step = jax.jit(train_step)


def powered_sum(x: jnp.ndarray, power: int) -> jnp.ndarray:
    return jnp.sum(x**power)


jitted_powered_sum = jax.jit(powered_sum, static_argnames=("power",))


def exercise_jitted_relu(x: jnp.ndarray) -> jnp.ndarray:
    """Exercise: implement jitted ReLU with jax.jit."""
    raise NotImplementedError("Implement this in solutions/lesson_04_solution.py")
