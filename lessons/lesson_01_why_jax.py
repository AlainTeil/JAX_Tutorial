"""Lesson 01: Why JAX for deep learning.

This lesson introduces JAX through a tiny logistic regression model written
with pure functions and immutable parameter updates.
"""

# Learning objectives:
# - Train a tiny logistic model with pure functions and immutable parameter updates.
# - Connect NumPy-shaped array code to JAX transformations and autodiff.
# - Read a before/after loss check as evidence that a functional update worked.
# Mental model:
# - JAX code is ordinary array code until a transformation stages it. Keep model state explicit so `grad`, `jit`, and tests see the same inputs.
# Common mistakes:
# - Treating parameter dictionaries like hidden mutable model state.
# - Interpreting a decreasing loss without checking shapes, labels, and logits.
# Recap:
# - The final loss is lower than the initial loss after repeated gradient steps.
# - Predictions have the same leading dimension as the labels.
# - The exercise check reports an accuracy between 0 and 1.

import jax
import jax.numpy as jnp


def make_toy_batch() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create a deterministic binary classification batch."""
    x = jnp.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    y = jnp.array([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)
    return x, y


def init_logreg_params(n_features: int) -> dict[str, jnp.ndarray]:
    """Initialize small deterministic parameters for reproducible lessons."""
    return {
        "w": jnp.zeros((n_features,), dtype=jnp.float32),
        "b": jnp.array(0.0, dtype=jnp.float32),
    }


def predict_logits(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x @ params["w"] + params["b"]


def predict_proba(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(predict_logits(params, x))


def binary_cross_entropy(proba: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    eps = 1e-7
    proba = jnp.clip(proba, eps, 1.0 - eps)
    return -jnp.mean(y * jnp.log(proba) + (1.0 - y) * jnp.log(1.0 - proba))


def loss_fn(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    proba = predict_proba(params, x)
    return binary_cross_entropy(proba, y)


def gd_step(
    params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray, lr: float
) -> tuple[dict[str, jnp.ndarray], jnp.ndarray]:
    """One gradient descent step using pure functional updates."""
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    new_params = {
        "w": params["w"] - lr * grads["w"],
        "b": params["b"] - lr * grads["b"],
    }
    return new_params, loss


def exercise_accuracy_from_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Exercise: implement binary accuracy computed from logits."""
    raise NotImplementedError("Implement this in solutions/lesson_01_solution.py")


if __name__ == "__main__":
    x_batch, y_batch = make_toy_batch()
    params_ = init_logreg_params(n_features=x_batch.shape[1])

    for _ in range(20):
        params_, current_loss = gd_step(params_, x_batch, y_batch, lr=0.2)

    print("Final loss:", float(current_loss))
    print("Probabilities:", predict_proba(params_, x_batch))
