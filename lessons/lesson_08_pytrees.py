"""Lesson 08: Working with pytrees."""

# Learning objectives:
# - Treat nested parameter containers as first-class JAX values.
# - Map, scale, add, flatten, and rebuild parameter pytrees.
# - Apply SGD updates to matching parameter and gradient structures.
# Mental model:
# - A pytree is the structure plus its array leaves. JAX transformations preserve and reason about both pieces.
# Common mistakes:
# - Updating only one leaf while forgetting the rest of the tree.
# - Flattening a tree and losing the treedef needed to rebuild it.
# Recap:
# - Tree arithmetic preserves the nested parameter structure.
# - Flattening plus unflattening round-trips the original tree.
# - Parameter counts equal the sum of every leaf size.

import jax
import jax.numpy as jnp

Params = dict[str, dict[str, jnp.ndarray] | jnp.ndarray]


def make_two_layer_params(in_dim: int, hidden_dim: int, out_dim: int) -> Params:
    return {
        "layer1": {
            "w": jnp.ones((in_dim, hidden_dim), dtype=jnp.float32),
            "b": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        },
        "layer2": {
            "w": jnp.ones((hidden_dim, out_dim), dtype=jnp.float32),
            "b": jnp.zeros((out_dim,), dtype=jnp.float32),
        },
    }


def tree_l2_norm(tree) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))


def tree_add(tree_a, tree_b):
    return jax.tree_util.tree_map(lambda a, b: a + b, tree_a, tree_b)


def tree_scale(tree, scalar: float):
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def sgd_update(params, grads, lr: float):
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def flatten_tree(tree):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    return leaves, treedef


def unflatten_tree(treedef, leaves):
    return jax.tree_util.tree_unflatten(treedef, leaves)


def exercise_count_parameters(tree) -> int:
    """Exercise: count scalar parameters in a pytree."""
    raise NotImplementedError("Implement this in solutions/lesson_08_solution.py")
