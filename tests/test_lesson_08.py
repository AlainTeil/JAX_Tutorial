import jax.numpy as jnp

from lessons.lesson_08_pytrees import (
    flatten_tree,
    make_two_layer_params,
    sgd_update,
    tree_add,
    tree_l2_norm,
    tree_scale,
    unflatten_tree,
)
from solutions.lesson_08_solution import count_parameters


def test_tree_l2_norm_positive():
    params = make_two_layer_params(in_dim=2, hidden_dim=3, out_dim=1)
    norm = tree_l2_norm(params)
    assert norm > 0.0


def test_tree_add_and_scale():
    params = make_two_layer_params(in_dim=2, hidden_dim=2, out_dim=1)
    doubled = tree_scale(params, 2.0)
    summed = tree_add(params, params)
    assert jnp.allclose(doubled["layer1"]["w"], summed["layer1"]["w"])


def test_sgd_update_changes_params():
    params = make_two_layer_params(in_dim=2, hidden_dim=2, out_dim=1)
    grads = tree_scale(params, 0.1)
    updated = sgd_update(params, grads, lr=0.5)
    assert not jnp.allclose(updated["layer1"]["w"], params["layer1"]["w"])


def test_flatten_unflatten_roundtrip_and_solution_count():
    params = make_two_layer_params(in_dim=2, hidden_dim=3, out_dim=1)
    leaves, treedef = flatten_tree(params)
    rebuilt = unflatten_tree(treedef, leaves)
    assert jnp.allclose(rebuilt["layer2"]["w"], params["layer2"]["w"])

    # Count is: 2*3 + 3 + 3*1 + 1 = 13
    assert count_parameters(params) == 13
