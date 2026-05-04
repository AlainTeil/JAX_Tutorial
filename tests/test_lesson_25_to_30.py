import jax
import jax.numpy as jnp

from lessons.lesson_25_transform_composition import (
    batch_loss,
    compiled_update,
    init_params,
    make_regression_batch,
    per_example_gradients,
    transform_order_summary,
)
from lessons.lesson_26_performance_memory import (
    blocking_sum,
    jitted_matmul_stack,
    make_benchmark_inputs,
    matmul_stack,
    tree_nbytes,
)
from lessons.lesson_27_data_input_pipelines import (
    iterate_minibatches,
    make_toy_records,
    standardize_from_train,
    train_val_split,
)
from lessons.lesson_28_multidevice_readiness import (
    collective_mode_label,
    device_inventory,
    required_global_batch_size,
    reshape_global_batch,
)
from lessons.lesson_29_experiment_analysis import (
    attach_environment,
    best_run,
    make_demo_runs,
    summarize_runs,
)
from lessons.lesson_30_deployment_model_cards import (
    calibration_bins,
    model_card,
    prediction_table,
    sigmoid_scores,
)
from solutions.lesson_25_solution import clip_gradients
from solutions.lesson_26_solution import estimate_batch_bytes
from solutions.lesson_27_solution import pad_to_batch
from solutions.lesson_28_solution import validate_global_batch_size
from solutions.lesson_29_solution import select_pareto_runs
from solutions.lesson_30_solution import apply_threshold


def test_transform_composition_update_and_gradient_clipping():
    x, y = make_regression_batch()
    params = init_params()
    grads = per_example_gradients(params, x, y)
    updated, loss = compiled_update(params, x, y, lr=0.1)

    assert grads["w"].shape == (x.shape[0], x.shape[1])
    assert set(updated) == set(params)
    assert float(batch_loss(updated, x, y)) < float(loss)
    assert "vmap_grad" in transform_order_summary()

    clipped = clip_gradients({"w": jnp.array([3.0, 4.0]), "b": jnp.array(0.0)}, max_norm=2.0)
    norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree_util.tree_leaves(clipped)))
    assert float(norm) <= 2.0001


def test_performance_memory_helpers():
    x, w = make_benchmark_inputs(4)
    eager = matmul_stack(x, w, depth=2)
    compiled = jitted_matmul_stack(x, w, depth=2)
    assert eager.shape == compiled.shape
    assert float(blocking_sum(compiled)) == float(jnp.sum(compiled))
    assert tree_nbytes({"x": x, "w": w}) == x.size * x.dtype.itemsize + w.size * w.dtype.itemsize
    assert estimate_batch_bytes((8, 4, 4), dtype_size=4) == 512


def test_data_input_pipeline_helpers():
    x, y = make_toy_records(10)
    (train_x, train_y), (val_x, val_y) = train_val_split(x, y, val_fraction=0.2)
    batches = iterate_minibatches(train_x, train_y, batch_size=3)
    train_standard, val_standard, mean, std = standardize_from_train(train_x, val_x)
    padded = pad_to_batch(x, batch_size=4, pad_value=-1.0)

    assert train_x.shape[0] == train_y.shape[0] == 8
    assert val_x.shape[0] == val_y.shape[0] == 2
    assert len(batches) == 3
    assert train_standard.shape == train_x.shape
    assert val_standard.shape == val_x.shape
    assert mean.shape == std.shape == (1, 2)
    assert padded.shape[0] % 4 == 0


def test_multidevice_readiness_helpers():
    info = device_inventory()
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    reshaped = reshape_global_batch(x, num_devices=2)
    local_device_count = info["local_device_count"]

    assert isinstance(local_device_count, int)
    assert local_device_count >= 1
    assert collective_mode_label() in {
        "single-device fallback path",
        "multi-device collective path",
    }
    assert required_global_batch_size(num_devices=2, per_device_batch=4) == 8
    assert reshaped.shape == (2, 2, 2)
    assert validate_global_batch_size(8, 2)
    assert not validate_global_batch_size(7, 2)


def test_experiment_analysis_helpers():
    runs = make_demo_runs()
    summary = summarize_runs(runs)
    enriched = attach_environment({"run_name": "demo"}, {"jax": "0.9"})
    pareto = select_pareto_runs(runs)

    assert best_run(runs)["run_name"] == "wider"
    assert summary["num_runs"] == 3
    assert enriched["packages"] == {"jax": "0.9"}
    assert [run["run_name"] for run in pareto] == ["wider", "fast"]


def test_deployment_model_card_helpers():
    logits = jnp.array([-2.0, -0.2, 0.4, 2.0], dtype=jnp.float32)
    probs = sigmoid_scores(logits)
    labels = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
    table = prediction_table(["a", "b", "c", "d"], probs, threshold=0.5)
    bins = calibration_bins(probs, labels, num_bins=2)
    card = model_card("toy", {"accuracy": 1.0}, ["toy data only"])

    assert apply_threshold(probs, threshold=0.5).tolist() == [0, 0, 1, 1]
    assert len(table) == 4
    assert sum(row["count"] for row in bins) == 4
    assert card["metrics"] == {"accuracy": 1.0}
    assert card["limitations"] == ["toy data only"]
