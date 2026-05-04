"""Lesson 23: JAX ecosystem tour (Optax, Flax, and related tools)."""

# Learning objectives:
# - Identify where core JAX ends and ecosystem libraries begin.
# - Run one Optax optimizer update by hand.
# - Use trend checks to validate training logs.
# Mental model:
# - JAX is the transformation and array substrate; libraries such as Flax and Optax package common modeling and optimization patterns on top.
# Common mistakes:
# - Treating ecosystem libraries as replacements for understanding JAX transformations.
# - Checking only the final loss without validating the trajectory direction.
# Recap:
# - The ecosystem summary names the role of each library.
# - An Optax update changes parameters and optimizer state.
# - Loss trend checks catch non-improving runs.

import flax.linen as nn
import jax.numpy as jnp
import optax


def ecosystem_summary() -> dict[str, str]:
    return {
        "jax": "Core transforms and array programming",
        "flax": "Neural network module system",
        "optax": "Gradient processing and optimizers",
        "orbax": "Checkpointing utilities",
        "equinox": "Alternative neural network library",
        "chex": "Testing and shape/type assertions",
    }


class TinyClassifier(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def one_optax_step(
    params: dict,
    grads: dict,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
) -> tuple[dict, optax.OptState]:
    updates, new_state = optimizer.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state


def demo_optax_adam_update() -> tuple[dict, dict]:
    params = {"w": jnp.array([1.0, -1.0], dtype=jnp.float32)}
    grads = {"w": jnp.array([0.5, -0.25], dtype=jnp.float32)}
    tx = optax.adam(learning_rate=0.1)
    state = tx.init(params)
    new_params, _ = one_optax_step(params, grads, tx, state)
    return params, new_params


def exercise_is_loss_decreasing(losses: jnp.ndarray) -> bool:
    """Exercise: decide whether final loss is lower than initial loss."""
    raise NotImplementedError("Implement this in solutions/lesson_23_solution.py")
