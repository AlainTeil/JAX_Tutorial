import jax
import jax.numpy as jnp

from lessons.lesson_19_pmap_parallel import (
    local_device_count,
    pmap_add_one,
    pmap_mean_across_devices,
    reshape_for_pmap,
)
from solutions.lesson_19_solution import data_parallel_sum


def test_reshape_for_pmap_and_add_one():
    ndev = local_device_count()
    x = jnp.arange(ndev * 2, dtype=jnp.float32)
    x_shaped = reshape_for_pmap(x)
    assert x_shaped.shape[0] == ndev

    y = pmap_add_one(x_shaped)
    assert jnp.allclose(y, x_shaped + 1.0)


def test_pmean_and_psum_solution():
    ndev = local_device_count()
    x = jnp.arange(ndev, dtype=jnp.float32).reshape(ndev, 1)

    mean_out = pmap_mean_across_devices(x)
    expected_mean = jnp.mean(x)
    assert jnp.allclose(mean_out, jnp.ones_like(mean_out) * expected_mean)

    sum_out = data_parallel_sum(x)
    expected_sum = jnp.sum(x)
    assert jnp.allclose(sum_out, jnp.ones_like(sum_out) * expected_sum)


def test_local_device_count_positive():
    assert local_device_count() >= 1
    assert len(jax.devices()) >= 1
