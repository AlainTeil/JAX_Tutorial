"""Lesson 28: Multi-device readiness checks."""

# Learning objectives:
# - Report device inventory before selecting a parallel execution strategy.
# - Validate global batch sizes for per-device splitting.
# - Label single-device fallback behavior separately from collective execution.
# Mental model:
# - Production multi-device code starts with readiness checks: device count, global batch divisibility, and explicit fallback behavior.
# Common mistakes:
# - Assuming a notebook is using collectives just because it imports `pmap` concepts.
# - Choosing a global batch size that cannot be split evenly across devices.
# Recap:
# - Device inventory reports backend and local device count.
# - Reshaped global batches expose a device-major leading axis.
# - Invalid batch/device combinations are rejected before parallel code runs.

import jax
import jax.numpy as jnp


def device_inventory() -> dict[str, object]:
    devices = jax.local_devices()
    return {
        "backend": jax.default_backend(),
        "local_device_count": len(devices),
        "device_kinds": tuple(device.device_kind for device in devices),
    }


def can_run_collectives() -> bool:
    return jax.local_device_count() > 1


def required_global_batch_size(num_devices: int, per_device_batch: int) -> int:
    return int(num_devices * per_device_batch)


def reshape_global_batch(x: jnp.ndarray, num_devices: int | None = None) -> jnp.ndarray:
    devices = jax.local_device_count() if num_devices is None else num_devices
    usable = (x.shape[0] // devices) * devices
    if usable == 0:
        raise ValueError("Input needs at least one example per selected device.")
    return x[:usable].reshape(devices, usable // devices, *x.shape[1:])


def collective_mode_label() -> str:
    if can_run_collectives():
        return "multi-device collective path"
    return "single-device fallback path"


def exercise_validate_global_batch_size(batch_size: int, num_devices: int) -> bool:
    """Exercise: validate that a global batch can be split evenly across devices."""
    raise NotImplementedError("Implement this in solutions/lesson_28_solution.py")
