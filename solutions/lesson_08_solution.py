"""Solutions for Lesson 08 exercises."""

import jax


def count_parameters(tree) -> int:
    leaves = jax.tree_util.tree_leaves(tree)
    return int(sum(x.size for x in leaves))
