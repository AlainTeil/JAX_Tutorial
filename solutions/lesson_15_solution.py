"""Solutions for Lesson 15 exercises."""

import jax


def parameter_count(params: dict) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))
