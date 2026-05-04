"""Solutions for Lesson 19 exercises."""

import jax
from jax import lax


def data_parallel_sum(x):
    fn = jax.pmap(lambda t: lax.psum(t, axis_name="d"), axis_name="d")
    return fn(x)
