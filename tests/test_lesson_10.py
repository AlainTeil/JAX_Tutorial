import jax.numpy as jnp

from lessons.lesson_10_optimization import quadratic_loss, run_gd, run_momentum
from solutions.lesson_10_solution import adaptive_lr


def test_quadratic_loss_minimum():
    optimum = jnp.array([3.0, -2.0], dtype=jnp.float32)
    assert jnp.isclose(quadratic_loss(optimum), 0.0)


def test_gd_and_momentum_converge():
    theta_gd, losses_gd = run_gd(num_steps=30, lr=0.2)
    theta_m, losses_m = run_momentum(num_steps=30, lr=0.2, beta=0.9)

    assert float(losses_gd[-1]) < float(losses_gd[0])
    assert float(losses_m[-1]) < float(losses_m[0])

    target = jnp.array([3.0, -2.0], dtype=jnp.float32)
    assert jnp.linalg.norm(theta_gd - target) < 1.0
    assert jnp.linalg.norm(theta_m - target) < 3.0


def test_solution_adaptive_lr_decreases():
    base = 0.1
    assert adaptive_lr(0, base) == base
    assert adaptive_lr(10, base) < adaptive_lr(1, base)
