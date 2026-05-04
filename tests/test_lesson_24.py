from pathlib import Path

import jax.numpy as jnp

from lessons.lesson_24_capstone import (
    evaluate,
    evaluation_report,
    load_capstone_checkpoint,
    make_xor_data,
    run_capstone_training,
    save_capstone_checkpoint,
)
from solutions.lesson_24_solution import confusion_matrix


def test_capstone_training_reduces_loss_and_improves_accuracy():
    state, losses, metrics = run_capstone_training(num_steps=300, learning_rate=0.1)
    assert float(losses[-1]) < float(losses[0])
    assert float(metrics["accuracy"]) >= 0.75

    x, y = make_xor_data()
    eval_metrics = evaluate(state, x, y)
    assert float(eval_metrics["accuracy"]) >= 0.75


def test_solution_confusion_matrix_counts():
    preds = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    labels = jnp.array([0, 1, 0, 0], dtype=jnp.int32)
    mat = confusion_matrix(preds, labels, num_classes=2)
    assert mat.shape == (2, 2)
    assert int(jnp.sum(mat)) == 4


def test_capstone_checkpoint_and_report(tmp_path: Path):
    state, _, _ = run_capstone_training(num_steps=50, learning_rate=0.1)
    x, y = make_xor_data()
    ckpt = tmp_path / "capstone_params.npz"
    save_capstone_checkpoint(ckpt, state.params)
    restored = load_capstone_checkpoint(ckpt)
    assert restored

    report = evaluation_report(state, x, y)
    assert set(report.keys()) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "confusion_matrix",
    }
    assert report["confusion_matrix"].shape == (2, 2)
