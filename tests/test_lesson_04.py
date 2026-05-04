import jax.numpy as jnp

from lessons.lesson_04_jit import (
    affine_forward,
    jitted_affine_forward,
    jitted_powered_sum,
    jitted_train_step,
    train_step,
)
from solutions.lesson_04_solution import jitted_relu


def test_jitted_and_eager_forward_match():
    params = {
        "w": jnp.array([[1.0], [2.0]], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    x = jnp.array([[3.0, 4.0]], dtype=jnp.float32)
    eager = affine_forward(params, x)
    compiled = jitted_affine_forward(params, x)
    assert jnp.allclose(eager, compiled)


def test_jitted_train_step_reduces_loss_like_eager():
    params = {
        "w": jnp.array([[0.0], [0.0]], dtype=jnp.float32),
        "b": jnp.array([0.0], dtype=jnp.float32),
    }
    x = jnp.array([[1.0, 1.0], [2.0, 2.0]], dtype=jnp.float32)
    y = jnp.array([[2.0], [4.0]], dtype=jnp.float32)

    p1, loss1 = train_step(params, x, y, 0.1)
    p2, loss2 = jitted_train_step(params, x, y, 0.1)

    assert jnp.allclose(loss1, loss2)
    assert jnp.allclose(p1["w"], p2["w"])
    assert jnp.allclose(p1["b"], p2["b"])


def test_jitted_powered_sum_static_arg():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    assert jnp.isclose(jitted_powered_sum(x, power=2), 14.0)
    assert jnp.isclose(jitted_powered_sum(x, power=3), 36.0)


def test_solution_jitted_relu():
    x = jnp.array([-2.0, 0.0, 3.0], dtype=jnp.float32)
    assert jnp.allclose(jitted_relu(x), jnp.array([0.0, 0.0, 3.0], dtype=jnp.float32))
