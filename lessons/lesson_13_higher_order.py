"""Lesson 13: Higher-order gradients and Jacobians in JAX."""

# Learning objectives:
# - Compose autodiff transforms for first and second derivatives.
# - Compute Jacobians and Hessians for small examples.
# - Use directional derivatives to ask local sensitivity questions.
# Mental model:
# - Higher-order derivatives are transformations of transformations. Keep functions small and scalar/vector contracts explicit.
# Common mistakes:
# - Building a Hessian for a large model when a vector product would answer the question.
# - Forgetting that higher-order derivatives amplify numerical and shape mistakes.
# Recap:
# - First and second derivatives match the scalar example.
# - Jacobian and Hessian shapes match input/output dimensions.
# - Directional derivatives return scalar sensitivity values.

import jax
import jax.numpy as jnp


def scalar_fn(x: jnp.ndarray) -> jnp.ndarray:
    return x**4 + 3.0 * x**2


def first_derivative(x: float) -> jnp.ndarray:
    return jax.grad(lambda t: scalar_fn(t))(x)


def second_derivative(x: float) -> jnp.ndarray:
    return jax.grad(jax.grad(lambda t: scalar_fn(t)))(x)


def vector_fn(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array(
        [
            x[0] ** 2 + x[1],
            jnp.sin(x[0]) + x[1] ** 3,
        ],
        dtype=jnp.float32,
    )


def jacobian_vector_fn(x: jnp.ndarray) -> jnp.ndarray:
    return jax.jacobian(vector_fn)(x)


def hessian_scalar_at(x: float) -> jnp.ndarray:
    return jax.hessian(lambda t: scalar_fn(t))(x)


def exercise_directional_derivative(x: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
    """Exercise: compute directional derivative of f(x)=sum(x^2)."""
    raise NotImplementedError("Implement this in solutions/lesson_13_solution.py")
