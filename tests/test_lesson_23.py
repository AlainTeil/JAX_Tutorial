import jax
import jax.numpy as jnp
import optax

from lessons.lesson_23_jax_ecosystem import (
    TinyClassifier,
    demo_optax_adam_update,
    ecosystem_summary,
    one_optax_step,
)
from solutions.lesson_23_solution import is_loss_decreasing


def test_ecosystem_summary_keys_present():
    summary = ecosystem_summary()
    for key in ["jax", "flax", "optax", "orbax", "equinox", "chex"]:
        assert key in summary


def test_optax_update_changes_params():
    old_params, new_params = demo_optax_adam_update()
    assert not jnp.allclose(old_params["w"], new_params["w"])


def test_flax_module_init_and_solution_loss_check():
    model = TinyClassifier(hidden_dim=4, out_dim=2)
    params = model.init(jax.random.key(0), jnp.ones((2, 3), dtype=jnp.float32))["params"]
    grads = jax.tree_util.tree_map(lambda p: jnp.ones_like(p) * 0.01, params)
    tx = optax.sgd(learning_rate=0.1)
    state = tx.init(params)
    new_params, _ = one_optax_step(params, grads, tx, state)
    leaves_old = jax.tree_util.tree_leaves(params)
    leaves_new = jax.tree_util.tree_leaves(new_params)
    assert any(not jnp.allclose(a, b) for a, b in zip(leaves_old, leaves_new, strict=True))

    losses = jnp.array([1.0, 0.8, 0.6], dtype=jnp.float32)
    assert is_loss_decreasing(losses)
