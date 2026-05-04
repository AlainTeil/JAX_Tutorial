"""Solutions for Lesson 21 exercises."""

import jax


def verify_named_sharding(x: jax.Array) -> bool:
    sharding = x.sharding
    return hasattr(sharding, "spec")
