import jax
import jax.numpy as jnp

from lessons.lesson_15_flax_intro import apply_model, build_model, init_params, mse_loss
from solutions.lesson_15_solution import parameter_count


def test_flax_init_and_apply_shapes():
    model = build_model(hidden_dim=4, out_dim=2)
    params = init_params(jax.random.key(0), model, input_shape=(3, 5))
    x = jnp.ones((3, 5), dtype=jnp.float32)
    logits = apply_model(model, params, x)
    assert logits.shape == (3, 2)


def test_mse_loss_nonnegative_and_parameter_count():
    model = build_model(hidden_dim=4, out_dim=2)
    params = init_params(jax.random.key(1), model, input_shape=(2, 5))
    x = jnp.ones((2, 5), dtype=jnp.float32)
    y = jnp.zeros((2, 2), dtype=jnp.float32)

    loss = mse_loss(model, params, x, y)
    assert loss >= 0.0
    assert parameter_count(params) > 0
