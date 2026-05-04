"""Lesson 19: Parallelization concepts with `jax.pmap`."""

# Learning objectives:
# - Inspect local devices before choosing a parallel strategy.
# - Reshape data for device-major `pmap` inputs.
# - Distinguish single-device fallbacks from true cross-device collectives.
# Mental model:
# - `pmap` maps one function over local devices. On a single device, collective operations are mathematically valid but do not demonstrate distributed communication.
# Common mistakes:
# - Mistaking a single-device fallback for evidence of scaling behavior.
# - Forgetting that `pmap` expects a leading axis sized by local device count.
# Recap:
# - The notebook reports the number of local devices before parallel examples.
# - Reshaped arrays have leading axis equal to local device count.
# - Single-device output is labeled as a fallback, not a scaling result.

import jax
import jax.numpy as jnp
from jax import lax


def local_device_count() -> int:
    return jax.local_device_count()


def is_multi_device() -> bool:
    return local_device_count() > 1


def reshape_for_pmap(x: jnp.ndarray) -> jnp.ndarray:
    ndev = local_device_count()
    usable = (x.shape[0] // ndev) * ndev
    if usable == 0:
        raise ValueError("Input needs at least local_device_count elements.")
    return x[:usable].reshape(ndev, usable // ndev, *x.shape[1:])


def pmap_add_one(x: jnp.ndarray) -> jnp.ndarray:
    if not is_multi_device():
        # CPU-first fallback: keep behavior explicit when only one device exists.
        return x + 1.0
    fn = jax.pmap(lambda t: t + 1.0)
    return fn(x)


def pmap_mean_across_devices(x: jnp.ndarray) -> jnp.ndarray:
    if not is_multi_device():
        # With one device, cross-device mean reduces to identity.
        return x
    fn = jax.pmap(lambda t: lax.pmean(t, axis_name="d"), axis_name="d")
    return fn(x)


def exercise_data_parallel_sum(x: jnp.ndarray) -> jnp.ndarray:
    """Exercise: sum values across mapped devices with `lax.psum`."""
    raise NotImplementedError("Implement this in solutions/lesson_19_solution.py")
