from pathlib import Path

import jax.numpy as jnp

from lessons.lesson_22_checkpointing_experiments import (
    checkpoint_params,
    load_run_metadata,
    make_run_metadata,
    restore_params,
    save_run_metadata,
)
from solutions.lesson_22_solution import best_run


def test_metadata_save_load_roundtrip(tmp_path: Path):
    meta = make_run_metadata("run-a", seed=0, learning_rate=0.1, num_steps=10)
    path = tmp_path / "meta.json"
    save_run_metadata(path, meta)
    loaded = load_run_metadata(path)
    assert loaded == meta


def test_checkpoint_roundtrip(tmp_path: Path):
    params = {
        "w": jnp.array([1.0, 2.0], dtype=jnp.float32),
        "b": jnp.array([0.5], dtype=jnp.float32),
    }
    path = tmp_path / "params.npz"
    checkpoint_params(path, params)
    restored = restore_params(path)
    assert jnp.allclose(restored["w"], params["w"])
    assert jnp.allclose(restored["b"], params["b"])


def test_solution_best_run():
    runs = [
        {"run_name": "a", "validation_loss": 0.7},
        {"run_name": "b", "validation_loss": 0.4},
        {"run_name": "c", "validation_loss": 0.5},
    ]
    assert best_run(runs)["run_name"] == "b"
