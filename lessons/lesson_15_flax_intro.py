"""Lesson 15: Introduction to Flax modules."""

# Learning objectives:
# - Define a compact Flax module for a small MLP.
# - Initialize and apply model parameters explicitly.
# - Measure loss without hiding params inside the module object.
# Mental model:
# - Flax modules describe computation; variables hold state. Initialization creates the parameter pytree that later calls consume.
# Common mistakes:
# - Calling a Flax module before initializing variables.
# - Assuming module instances own trained parameters like mutable objects.
# Recap:
# - Initialized parameter collections contain a `params` tree.
# - Model outputs have batch dimension and output dimension.
# - Parameter count is positive and matches the tree leaves.

import flax.linen as nn
import jax
import jax.numpy as jnp


class TinyMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, name="dense_hidden")(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim, name="dense_out")(x)
        return x


def build_model(hidden_dim: int = 8, out_dim: int = 2) -> TinyMLP:
    return TinyMLP(hidden_dim=hidden_dim, out_dim=out_dim)


def init_params(key: jax.Array, model: TinyMLP, input_shape: tuple[int, ...]) -> dict:
    dummy_x = jnp.zeros(input_shape, dtype=jnp.float32)
    variables = model.init(key, dummy_x)
    return variables["params"]


def apply_model(model: TinyMLP, params: dict, x: jnp.ndarray) -> jnp.ndarray:
    return model.apply({"params": params}, x)


def mse_loss(model: TinyMLP, params: dict, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    pred = apply_model(model, params, x)
    return jnp.mean((pred - y) ** 2)


def exercise_parameter_count(params: dict) -> int:
    """Exercise: count total number of trainable parameters."""
    raise NotImplementedError("Implement this in solutions/lesson_15_solution.py")
