"""Generate executable, output-bearing workbook notebooks."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import textwrap
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from course.curriculum import LESSONS, LessonSpec  # noqa: E402

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3.12",
    },
}

DEMO_CODE: dict[int, str] = {
    1: """
        x, y = lesson.make_toy_batch()
        params = lesson.init_logreg_params(x.shape[1])
        initial_loss = lesson.loss_fn(params, x, y)
        for _ in range(30):
            params, _ = lesson.gd_step(params, x, y, lr=0.2)
        final_loss = lesson.loss_fn(params, x, y)
        logits = lesson.predict_logits(params, x)
        print(f'shape={x.shape}, initial_loss={float(initial_loss):.4f}, final_loss={float(final_loss):.4f}')
        print('exercise_accuracy=', float(solution.accuracy_from_logits(logits, y)))
    """,
    2: """
        arrays = lesson.create_core_arrays()
        vector, matrix = arrays['vector'], arrays['matrix']
        updated = lesson.immutable_set(vector, 1, 99.0)
        normalized = solution.row_normalize(jnp.array([[1.0, 1.0], [2.0, 6.0]], dtype=jnp.float32))
        print('vector=', vector.tolist(), 'updated=', updated.tolist())
        print('matrix_shape=', matrix.shape, 'row_sums=', jnp.sum(normalized, axis=1).tolist())
    """,
    3: """
        params = {'w': jnp.array([1.0, -1.0]), 'b': jnp.array(0.1)}
        x = jnp.array([[1.0, 2.0], [2.0, -1.0]], dtype=jnp.float32)
        y = jnp.array([0.0, 1.0], dtype=jnp.float32)
        _, grads = lesson.mse_grads(params, x, y)
        _, mae_grads = solution.mae_grads(params, x, y)
        print('scalar_grad_at_3=', float(lesson.scalar_grad(3.0)))
        print('mse_grad_shapes=', {k: v.shape for k, v in grads.items()})
        print('mae_grad_w=', mae_grads['w'].round(4).tolist())
    """,
    4: """
        params = {'w': jnp.array([[1.0], [-1.0]]), 'b': jnp.array([0.0])}
        x = jnp.array([[1.0, 2.0], [2.0, 1.0]], dtype=jnp.float32)
        y = jnp.array([[0.0], [1.0]], dtype=jnp.float32)
        new_params, loss = lesson.train_step(params, x, y, lr=0.1)
        relu_values = solution._relu(jnp.array([-2.0, 0.0, 3.0]))
        print('loss=', float(loss), 'updated_keys=', sorted(new_params))
        print('powered_sum=', float(lesson.powered_sum(jnp.array([2.0, 3.0]), 2)))
        print('relu=', relu_values.tolist())
    """,
    5: """
        batch = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        loop = lesson.batch_squared_l2_loop(batch)
        mapped = lesson.batch_squared_l2_vmap(batch)
        cosine = solution.batch_cosine_similarity(batch, batch)
        print('loop_vs_vmap=', jnp.allclose(loop, mapped).item(), loop.tolist())
        print('cosine_self=', cosine.round(4).tolist())
    """,
    6: """
        xs = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        print('safe_inverse=', [float(lesson.safe_inverse(v)) for v in (0.0, 2.0)])
        print('scan_sum=', lesson.cumulative_sum_scan(xs).tolist())
        print('weighted=', solution.weighted_cumsum(xs, alpha=0.5).round(4).tolist())
    """,
    7: """
        key = lesson.make_key(0)
        k1, k2, _ = lesson.split_three(key)
        sample = lesson.sample_normal(k1, (3,))
        dropped = lesson.dropout(k2, jnp.ones((6,)), keep_prob=0.5)
        mean, std = solution.gaussian_stats(lesson.make_key(1), 1000)
        print('sample=', sample.round(4).tolist())
        print('dropout_shape=', dropped.shape, 'stats=', (round(float(mean), 3), round(float(std), 3)))
    """,
    8: """
        params = lesson.make_two_layer_params(3, 4, 2)
        leaves, treedef = lesson.flatten_tree(params)
        rebuilt = lesson.unflatten_tree(treedef, leaves)
        print('leaf_count=', len(leaves), 'l2_norm=', round(float(lesson.tree_l2_norm(params)), 4))
        print('round_trip_keys=', sorted(rebuilt.keys()))
        print('parameter_count=', int(solution.count_parameters(params)))
    """,
    9: """
        x, y = lesson.make_regression_data()
        params, losses = lesson.train_linear_model(num_steps=30, lr=0.1)
        mae = solution.mae_loss(params, x, y)
        print('loss_start_end=', (round(float(losses[0]), 4), round(float(losses[-1]), 4)))
        print('prediction_shape=', lesson.linear_forward(params, x).shape, 'mae=', round(float(mae), 4))
    """,
    10: """
        _, gd_losses = lesson.run_gd(num_steps=8, lr=0.1)
        _, mom_losses = lesson.run_momentum(num_steps=8, lr=0.1, beta=0.9)
        lr0 = solution.adaptive_lr(0, 0.1)
        lr10 = solution.adaptive_lr(10, 0.1)
        print('gd_last=', round(float(gd_losses[-1]), 4), 'momentum_last=', round(float(mom_losses[-1]), 4))
        print('adaptive_lr=', (round(float(lr0), 4), round(float(lr10), 4)))
    """,
    11: """
        key = jax.random.key(0)
        x, y = lesson.make_classification_data(12)
        xb, yb = lesson.get_batch(x, y, 0, 4)
        shuffled_x, shuffled_y = lesson.shuffle_data(key, x, y)
        mean_loss = solution.epoch_mean_loss(jnp.array([0.9, 0.6, 0.3]))
        print('batch_shape=', xb.shape, yb.shape, 'indices=', lesson.batch_indices(12, 5))
        print('shuffle_preserves_shape=', shuffled_x.shape == x.shape and shuffled_y.shape == y.shape)
        print('epoch_mean_loss=', round(float(mean_loss), 4))
    """,
    12: """
        logits = jnp.array([[1000.0, 1001.0, 999.0]], dtype=jnp.float32)
        labels = jnp.array([1], dtype=jnp.int32)
        one_hot = jax.nn.one_hot(labels, 3)
        smooth = solution.label_smoothing(one_hot, epsilon=0.1)
        print('stable_softmax=', lesson.stable_softmax(logits).round(4).tolist())
        print('cross_entropy=', round(float(lesson.cross_entropy_with_logits(logits, labels)), 4))
        print('smoothed_row_sum=', float(jnp.sum(smooth)))
    """,
    13: """
        x = jnp.array([1.0, 2.0], dtype=jnp.float32)
        direction = jnp.array([0.5, -0.25], dtype=jnp.float32)
        print('first_second=', (round(float(lesson.first_derivative(2.0)), 4), round(float(lesson.second_derivative(2.0)), 4)))
        print('jacobian_shape=', lesson.jacobian_vector_fn(x).shape)
        print('directional=', round(float(solution.directional_derivative(x, direction)), 4))
    """,
    14: """
        x = jnp.array([1.0, jnp.nan, 3.0, jnp.nan])
        jaxpr = lesson.polynomial_jaxpr(jnp.ones((2,)))
        print('has_nans=', bool(lesson.has_nans(x)), 'count=', int(solution.count_nans(x)))
        print('jaxpr_contains=', str(jaxpr).splitlines()[0][:60])
        print('finite_diff=', round(float(lesson.finite_difference_grad(2.0)), 4))
    """,
    15: """
        key = jax.random.key(0)
        model = lesson.build_model(hidden_dim=4, out_dim=2)
        params = lesson.init_params(key, model, (1, 3))
        out = lesson.apply_model(model, params, jnp.ones((2, 3)))
        print('output_shape=', out.shape)
        print('parameter_count=', int(solution.parameter_count(params)))
    """,
    16: """
        state, losses, x, y = lesson.train_for_steps(num_steps=20, learning_rate=0.1)
        logits = state.apply_fn({'params': state.params}, x)
        acc = solution.classification_accuracy(logits, y)
        print('loss_start_end=', (round(float(losses[0]), 4), round(float(losses[-1]), 4)))
        print('accuracy=', round(float(acc), 4), 'state_step=', int(state.step))
    """,
    17: """
        params = {'w': jnp.array([[1.0], [-1.0]]), 'b': jnp.array([0.0])}
        x = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)
        y = jnp.array([1.0, 0.0], dtype=jnp.float32)
        logits = lesson.linear_logits(params, x)
        precision, recall = lesson.precision_recall_from_logits(logits, y)
        print('total_loss=', round(float(lesson.total_loss(params, x, y, weight_decay=0.1)), 4))
        print('precision_recall_f1=', (round(float(precision), 4), round(float(recall), 4), round(float(solution.f1_score(precision, recall)), 4)))
    """,
    18: """
        xs = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
        params = {'wh': jnp.array([[0.5]]), 'wx': jnp.array([[1.0]]), 'b': jnp.array([0.0])}
        outputs = lesson.run_simple_rnn(params, jnp.array([0.0]), xs.reshape(-1, 1))
        print('scan_sum=', lesson.cumulative_sum_scan(xs).tolist())
        print('rnn_output_shape=', outputs.shape)
        print('reverse_suffix=', solution.reverse_scan_suffix_sum(xs).tolist())
    """,
    19: """
        devices = lesson.local_device_count()
        x = jnp.arange(max(devices, 1) * 2, dtype=jnp.float32)
        reshaped = lesson.reshape_for_pmap(x)
        added = lesson.pmap_add_one(reshaped)
        summed = solution.data_parallel_sum(reshaped)
        print('local_devices=', devices, 'multi_device=', lesson.is_multi_device())
        print('reshaped=', reshaped.shape, 'added_shape=', added.shape)
        print('parallel_sum_or_fallback_shape=', getattr(summed, 'shape', ()))
    """,
    20: """
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        w = jnp.ones((3, 2), dtype=jnp.float32)
        y = lesson.pjit_matmul(x, w)
        placed = solution.make_sharded_array(x)
        print('devices=', len(jax.devices()), 'matmul=', y.tolist())
        print('sharding_type=', type(lesson.inspect_array_sharding(placed)).__name__)
    """,
    21: """
        x = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        placed = lesson.place_sharded_batch(x)
        ok = solution.verify_named_sharding(placed)
        print('devices=', len(jax.devices()), 'placed_shape=', placed.shape)
        print('sharding=', type(placed.sharding).__name__, 'has_data_axis=', bool(ok))
    """,
    22: """
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'demo_meta.json'
            meta = lesson.make_run_metadata('demo-run', seed=0, learning_rate=0.1, num_steps=100)
            lesson.save_run_metadata(path, meta)
            restored = lesson.load_run_metadata(path)
        runs = [{'name': 'a', 'validation_loss': 0.4}, {'name': 'b', 'validation_loss': 0.2}]
        print('metadata_name=', restored['run_name'], 'steps=', restored['num_steps'])
        print('best_run=', solution.best_run(runs)['name'])
    """,
    23: """
        summary = lesson.ecosystem_summary()
        params, new_params = lesson.demo_optax_adam_update()
        losses = jnp.array([1.0, 0.8, 0.5], dtype=jnp.float32)
        print('libraries=', sorted(summary.keys()))
        print('params_changed=', bool(jnp.any(params['w'] != new_params['w'])))
        print('loss_decreasing=', bool(solution.is_loss_decreasing(losses)))
    """,
    24: """
        state, losses, metrics = lesson.run_capstone_training(num_steps=120, learning_rate=0.1)
        x, y = lesson.make_xor_data()
        report = lesson.evaluation_report(state, x, y)
        preds = jnp.argmax(state.apply_fn({'params': state.params}, x), axis=-1)
        conf = solution.confusion_matrix(preds, y, num_classes=2)
        print('loss_start_end=', (round(float(losses[0]), 4), round(float(losses[-1]), 4)))
        print('accuracy=', round(float(metrics['accuracy']), 4), 'report_keys=', sorted(report.keys()))
        print('confusion_sum=', int(jnp.sum(conf)), 'matrix=', conf.tolist())
    """,
    25: """
        x, y = lesson.make_regression_batch()
        params = lesson.init_params()
        grads = lesson.per_example_gradients(params, x, y)
        updated, loss = lesson.compiled_update(params, x, y, lr=0.1)
        clipped = solution.clip_gradients({'w': jnp.array([3.0, 4.0]), 'b': jnp.array(0.0)}, max_norm=2.0)
        print('loss=', round(float(loss), 4), 'updated_keys=', sorted(updated.keys()))
        print('per_example_grad_shapes=', {k: v.shape for k, v in grads.items()})
        print('clipped_norm=', round(float(jnp.sqrt(sum(jnp.sum(v ** 2) for v in clipped.values()))), 4))
    """,
    26: """
        x, w = lesson.make_benchmark_inputs(4)
        eager = lesson.matmul_stack(x, w, depth=2)
        compiled, elapsed = lesson.timed_blocking_run(lesson.jitted_matmul_stack, x, w, depth=2)
        bytes_estimate = solution.estimate_batch_bytes((8, 4, 4), dtype_size=4)
        print('same_shape=', eager.shape == compiled.shape, 'elapsed_nonnegative=', elapsed >= 0.0)
        print('tree_nbytes=', lesson.tree_nbytes({'x': x, 'w': w}))
        print('batch_bytes=', bytes_estimate)
    """,
    27: """
        x, y = lesson.make_toy_records(10)
        (train_x, train_y), (val_x, val_y) = lesson.train_val_split(x, y, val_fraction=0.2)
        batches = lesson.iterate_minibatches(train_x, train_y, batch_size=3)
        _, val_standard, mean, std = lesson.standardize_from_train(train_x, val_x)
        padded = solution.pad_to_batch(x, batch_size=4, pad_value=-1.0)
        print('split_shapes=', train_x.shape, val_x.shape, train_y.shape, val_y.shape)
        print('num_batches=', len(batches), 'mean_shape=', mean.shape, 'std_min=', round(float(jnp.min(std)), 4))
        print('padded_shape=', padded.shape, 'val_standard_shape=', val_standard.shape)
    """,
    28: """
        info = lesson.device_inventory()
        x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
        reshaped = lesson.reshape_global_batch(x, num_devices=1)
        valid = solution.validate_global_batch_size(batch_size=8, num_devices=2)
        invalid = solution.validate_global_batch_size(batch_size=7, num_devices=2)
        print('inventory=', info['backend'], info['local_device_count'], lesson.collective_mode_label())
        print('reshaped_shape=', reshaped.shape)
        print('valid_invalid=', bool(valid), bool(invalid))
    """,
    29: """
        runs = lesson.make_demo_runs()
        ranked = lesson.rank_runs(runs)
        summary = lesson.summarize_runs(runs)
        enriched = lesson.attach_environment({'run_name': 'demo'}, {'jax': jax.__version__})
        pareto = solution.select_pareto_runs(runs)
        print('best=', ranked[0]['run_name'], 'summary_best=', summary['best_run_name'])
        print('packages=', enriched['packages'])
        print('pareto=', [run['run_name'] for run in pareto])
    """,
    30: """
        logits = jnp.array([-2.0, -0.2, 0.4, 2.0], dtype=jnp.float32)
        probs = lesson.sigmoid_scores(logits)
        labels = jnp.array([0, 0, 1, 1], dtype=jnp.int32)
        table = lesson.prediction_table(['a', 'b', 'c', 'd'], probs, threshold=0.5)
        bins = lesson.calibration_bins(probs, labels, num_bins=2)
        card = lesson.model_card('toy classifier', {'accuracy': 1.0}, ['toy data only'])
        preds = solution.apply_threshold(probs, threshold=0.5)
        print('preds=', preds.tolist(), 'table_rows=', len(table))
        print('bins=', bins)
        print('card_keys=', sorted(card.keys()))
    """,
}


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code_cell(source: str, execution_count: int, outputs: list[dict] | None = None) -> dict:
    clean = textwrap.dedent(source).strip() + "\n"
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs or [],
        "source": [line + "\n" for line in clean.splitlines()],
    }


def execute_cells(cells: list[dict], notebook_path: Path) -> list[dict]:
    env: dict[str, object] = {"__name__": "__main__", "__file__": str(notebook_path)}
    executed: list[dict] = []
    count = 1
    for cell in cells:
        if cell["cell_type"] != "code":
            executed.append(cell)
            continue
        source = "".join(cell["source"])
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(compile(source, str(notebook_path), "exec"), env, env)
        except Exception:
            buffer.write(traceback.format_exc())
            raise
        output_text = buffer.getvalue()
        outputs = []
        if output_text:
            outputs.append(
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": output_text.splitlines(keepends=True),
                }
            )
        executed.append(code_cell(source, execution_count=count, outputs=outputs))
        count += 1
    return executed


def build_notebook(lesson: LessonSpec) -> dict:
    number = f"{lesson.number:02d}"
    cells = [
        markdown_cell(
            f"""
            # Lesson {number}: {lesson.title}

            **Phase:** {lesson.phase}  
            **Script:** `{lesson.module.replace(".", "/")}.py`  
            **Exercise:** `{lesson.exercise_name}`  
            **Reference solution:** `{lesson.solution}.{lesson.solution_function}`
            """
        ),
        markdown_cell(
            "## What This Lesson Teaches\n"
            + "\n".join(f"- {objective}" for objective in lesson.objectives)
            + "\n\n## Mental Model\n"
            + lesson.mental_model
        ),
        code_cell(
            f"""
            from pathlib import Path

            import jax
            import jax.numpy as jnp

            import {lesson.module} as lesson
            import {lesson.solution} as solution
            """,
            execution_count=0,
        ),
        markdown_cell(
            "## Guided Demo\n"
            "Run the cell below before attempting the exercise. It prints compact evidence for the key shapes, values, or state transitions."
        ),
        code_cell(DEMO_CODE[lesson.number], execution_count=0),
        markdown_cell(
            "## Workbook Exercise\n"
            f"TODO: `{lesson.exercise_name}`. {lesson.exercise_prompt}\n\n"
            f"Hint: {lesson.exercise_hint}\n\n"
            "The next cell prints the exercise name, reference solution path, and ready status so this published notebook stays executable. Replace it with your own scratch implementation while studying."
        ),
        code_cell(
            f"""
            print('exercise:', '{lesson.exercise_name}')
            print('reference:', '{lesson.solution}.{lesson.solution_function}')
            print('status: ready for student implementation')
            """,
            execution_count=0,
        ),
        markdown_cell(
            "## Expected Output Checkpoint\n" + "\n".join(f"- {item}" for item in lesson.checkpoint)
        ),
        markdown_cell(
            "## Common Mistakes\n"
            + "\n".join(f"- {mistake}" for mistake in lesson.common_mistakes)
            + "\n\n## Extension\n"
            + lesson.extension
        ),
    ]
    notebook_path = ROOT / lesson.notebook
    return {
        "cells": execute_cells(cells, notebook_path),
        "metadata": KERNEL_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    for lesson in LESSONS:
        notebook = build_notebook(lesson)
        path = ROOT / lesson.notebook
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
