"""Lesson 10: Optimization basics with gradient descent."""

# Learning objectives:
# - Minimize a convex quadratic with gradient descent.
# - Track optimizer state for momentum updates.
# - Schedule learning rates as explicit functions of step number.
# Mental model:
# - Optimizers are state machines over parameters and auxiliary state. Keeping both explicit makes them easy to test and replace.
# Common mistakes:
# - Treating momentum velocity as hidden mutable state.
# - Judging an optimizer without plotting or inspecting the loss trajectory.
# Recap:
# - Gradient descent and momentum both move the parameter toward the quadratic minimum.
# - Momentum carries velocity between steps.
# - The adaptive learning rate remains positive and decreases over time.

import jax
import jax.numpy as jnp


def quadratic_loss(theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((theta - jnp.array([3.0, -2.0], dtype=jnp.float32)) ** 2)


def gd_update(theta: jnp.ndarray, lr: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    loss, grad = jax.value_and_grad(quadratic_loss)(theta)
    return theta - lr * grad, loss


def momentum_update(
    theta: jnp.ndarray, velocity: jnp.ndarray, lr: float, beta: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    loss, grad = jax.value_and_grad(quadratic_loss)(theta)
    velocity = beta * velocity + (1.0 - beta) * grad
    theta = theta - lr * velocity
    return theta, velocity, loss


def run_gd(num_steps: int = 40, lr: float = 0.2):
    theta = jnp.array([10.0, 10.0], dtype=jnp.float32)
    losses = []
    for _ in range(num_steps):
        theta, loss = gd_update(theta, lr)
        losses.append(loss)
    return theta, jnp.array(losses)


def run_momentum(num_steps: int = 40, lr: float = 0.3, beta: float = 0.9):
    theta = jnp.array([10.0, 10.0], dtype=jnp.float32)
    velocity = jnp.zeros_like(theta)
    losses = []
    for _ in range(num_steps):
        theta, velocity, loss = momentum_update(theta, velocity, lr, beta)
        losses.append(loss)
    return theta, jnp.array(losses)


def exercise_adaptive_lr(step: int, base_lr: float) -> float:
    """Exercise: implement a simple inverse-time learning rate schedule."""
    raise NotImplementedError("Implement this in solutions/lesson_10_solution.py")
