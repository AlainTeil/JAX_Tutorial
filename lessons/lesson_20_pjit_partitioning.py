"""Lesson 20: Partitioning and `pjit` concepts."""

# Learning objectives:
# - Build a named device mesh for partitioning examples.
# - Attach replicated and first-dimension shardings to arrays.
# - Understand single-device partitioning as placement metadata rather than speedup.
# Mental model:
# - Named sharding describes how logical array axes map to a device mesh. The same API can be demonstrated on one device, but partitioning is only visible with multiple devices.
# Common mistakes:
# - Equating `NamedSharding` metadata with actual multi-device parallelism on one device.
# - Using mesh contexts mechanically without explaining which calls need them.
# Recap:
# - Matrix multiplication returns the same numerical result as ordinary `x @ w`.
# - Sharding inspection returns a real sharding object.
# - The notebook states whether the current run is single-device or multi-device.

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def make_mesh() -> Mesh:
    devices = np.array(jax.devices())
    return Mesh(devices, axis_names=("data",))


def replicate_sharding(mesh: Mesh) -> NamedSharding:
    return NamedSharding(mesh, P())


def shard_first_dim(mesh: Mesh) -> NamedSharding:
    # On a single device this still behaves correctly but does not provide real partitioning.
    return NamedSharding(mesh, P("data", None))


def pjit_matmul(x: jnp.ndarray, w: jnp.ndarray) -> jnp.ndarray:
    if len(jax.devices()) == 1:
        # CPU-first fallback: same math without requiring multi-device semantics.
        return x @ w
    with make_mesh() as mesh:
        rep = replicate_sharding(mesh)
        compiled = jax.jit(
            lambda a, b: a @ b,
            in_shardings=(rep, rep),
            out_shardings=rep,
        )
        return compiled(x, w)


def inspect_array_sharding(x: jnp.ndarray):
    return x.sharding


def exercise_make_sharded_array(x: jnp.ndarray) -> jax.Array:
    """Exercise: place an array on a named sharding."""
    raise NotImplementedError("Implement this in solutions/lesson_20_solution.py")
