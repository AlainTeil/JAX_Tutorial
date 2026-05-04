"""Lesson 03: Automatic differentiation in JAX."""

# Learning objectives:
# - Differentiate scalar-valued functions with `jax.grad`.
# - Compute parameter gradients for a linear model loss.
# - Inspect Jacobians when a function returns vectors instead of scalars.
# Mental model:
# - Reverse-mode autodiff needs a scalar objective. When outputs are structured, choose the derivative object deliberately: gradient, Jacobian, or Hessian.
# Common mistakes:
# - Calling `grad` on a vector-valued function and expecting a full Jacobian.
# - Differentiating through integer or boolean values instead of floating arrays.
# Recap:
# - The scalar derivative matches the analytic slope at the demo point.
# - Gradient pytrees match the parameter pytree structure.
# - The MAE gradient exposes the sign-based behavior of absolute error.

import jax
import jax.numpy as jnp


def scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
    return x**3 + 2.0 * x


def scalar_grad(x: float) -> jnp.ndarray:
    return jax.grad(lambda t: scalar_fn(t))(x)


def second_derivative(x: float) -> jnp.ndarray:
    return jax.grad(jax.grad(lambda t: scalar_fn(t)))(x)


def linear_model(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return x * params["w"] + params["b"]


def mse_loss(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = linear_model(params, x)
    return jnp.mean((preds - y) ** 2)


def mse_grads(
    params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    return jax.value_and_grad(mse_loss)(params, x, y)


def softmax_jacobian(logits: jnp.ndarray) -> jnp.ndarray:
    def softmax_fn(t: jnp.ndarray) -> jnp.ndarray:
        return jax.nn.softmax(t)

    return jax.jacobian(softmax_fn)(logits)


def exercise_mae_grad(params: dict[str, jnp.ndarray], x: jnp.ndarray, y: jnp.ndarray):
    """Exercise: compute gradients of MAE loss."""
    raise NotImplementedError("Implement this in solutions/lesson_03_solution.py")
