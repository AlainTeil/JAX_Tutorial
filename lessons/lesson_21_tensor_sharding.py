"""Lesson 21: Tensor sharding with named axes."""

# Learning objectives:
# - Place arrays with replicated and data-axis shardings.
# - Inspect sharding metadata after device placement.
# - Run linear algebra while preserving placement semantics.
# Mental model:
# - Sharding is part of an array's placement contract. Correct code should make the intended axis names visible and testable.
# Common mistakes:
# - Assuming sharding is visible from array values alone.
# - Forgetting to verify axis names when teaching partitioned tensor layouts.
# Recap:
# - Placed arrays report sharding metadata.
# - Single-device examples still preserve named-axis intent.
# - The verification exercise returns a boolean-style result.

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def make_data_mesh() -> Mesh:
    devices = np.array(jax.devices())
    return Mesh(devices, axis_names=("data",))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def data_axis_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P("data", None))


def place_replicated(x: jnp.ndarray) -> jax.Array:
    with make_data_mesh() as mesh:
        sharding = replicated_sharding(mesh)
        return jax.device_put(x, sharding)


def place_sharded_batch(x: jnp.ndarray) -> jax.Array:
    with make_data_mesh() as mesh:
        # On single-device setups this is effectively replicated placement.
        sharding = data_axis_sharding(mesh)
        return jax.device_put(x, sharding)


def sharded_linear(x: jnp.ndarray, w: jnp.ndarray) -> jax.Array:
    if len(jax.devices()) == 1:
        # CPU-first fallback keeps semantics clear when true sharding is unavailable.
        return x @ w
    with make_data_mesh() as mesh:
        in_sharding_x = data_axis_sharding(mesh)
        in_sharding_w = replicated_sharding(mesh)
        out_sharding = data_axis_sharding(mesh)
        compiled = jax.jit(
            lambda a, b: a @ b,
            in_shardings=(in_sharding_x, in_sharding_w),
            out_shardings=out_sharding,
        )
        return compiled(x, w)


def exercise_verify_sharding_name(x: jax.Array) -> bool:
    """Exercise: verify the array has a named sharding."""
    raise NotImplementedError("Implement this in solutions/lesson_21_solution.py")
