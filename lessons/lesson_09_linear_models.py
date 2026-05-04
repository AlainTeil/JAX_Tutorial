"""Lesson 09: Linear models from scratch in JAX."""

# Learning objectives:
# - Build a deterministic linear regression dataset.
# - Train a linear model with explicit parameters and gradients.
# - Compare MSE and MAE as scalar training objectives.
# Mental model:
# - A model can be just a pure function plus an explicit parameter pytree. Frameworks add ergonomics, not magic.
# Common mistakes:
# - Mixing data generation, model evaluation, and optimization into one opaque function.
# - Changing loss definitions without checking their gradient behavior.
# Recap:
# - Training loss decreases over repeated gradient steps.
# - Learned parameters produce predictions with the target shape.
# - MAE is non-negative and shares the batch axis with MSE examples.

import jax
import jax.numpy as jnp


def make_regression_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.array([[0.0], [1.0], [2.0], [3.0]], dtype=jnp.float32)
    y = 2.0 * x + 1.0
    return x, y


def init_linear_params() -> dict[str, jnp.ndarray]:
    return {
        "w": jnp.array([[0.0]], dtype=jnp.float32),
        "b": jnp.array([0.0], dtype=jnp.float32),
    }


def linear_forward(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def mse_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = linear_forward(params, x)
    return jnp.mean((pred - y) ** 2)


def train_step(
    params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    loss, grads = jax.value_and_grad(mse_loss)(params, x, y)
    new_params = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"],
    }
    return new_params, loss


def train_linear_model(num_steps: int = 100, lr: float = 0.1):
    x, y = make_regression_data()
    params = init_linear_params()
    losses = []
    for _ in range(num_steps):
        params, loss = train_step(params, x, y, lr)
        losses.append(loss)
    return params, jnp.array(losses)


def exercise_mae_loss(
    params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray
) -> jnp.ndarray:
    """Exercise: implement MAE loss for linear regression."""
    raise NotImplementedError("Implement this in solutions/lesson_09_solution.py")
