"""Solutions for Lesson 25 exercises."""

import jax
import jax.numpy as jnp


def clip_gradients(grads: dict[str, jnp.ndarray], max_norm: float) -> dict[str, jnp.ndarray]:
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
    scale = jnp.minimum(1.0, max_norm / jnp.maximum(norm, 1e-12))
    return jax.tree_util.tree_map(lambda leaf: leaf * scale, grads)
