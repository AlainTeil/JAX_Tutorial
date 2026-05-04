"""Solutions for Lesson 20 exercises."""

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from lessons.lesson_20_pjit_partitioning import make_mesh


def make_sharded_array(x):
    with make_mesh() as mesh:
        sharding = NamedSharding(mesh, P())
        return jax.device_put(x, sharding)
