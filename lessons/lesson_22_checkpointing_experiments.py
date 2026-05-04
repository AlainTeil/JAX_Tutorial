"""Lesson 22: Checkpointing and experiment management basics."""

# Learning objectives:
# - Record run metadata in plain serializable structures.
# - Save and restore small parameter pytrees.
# - Keep notebook-generated artifacts out of tracked repository files.
# Mental model:
# - Experiment state is useful only when it can be restored and compared. Keep metadata explicit and write demos to temporary paths.
# Common mistakes:
# - Writing notebook demo outputs over tracked files.
# - Saving parameters without enough metadata to reproduce the run.
# Recap:
# - Metadata round-trips through JSON without changing fields.
# - Checkpointed parameter leaves restore with matching values.
# - Notebook execution writes only to a temporary directory.

import json
from pathlib import Path

import jax.numpy as jnp


def make_run_metadata(run_name: str, seed: int, learning_rate: float, num_steps: int) -> dict:
    return {
        "run_name": run_name,
        "seed": int(seed),
        "learning_rate": float(learning_rate),
        "num_steps": int(num_steps),
    }


def save_run_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def load_run_metadata(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_params(path: Path, params: dict[str, jnp.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_params = {k: jnp.asarray(v) for k, v in params.items()}
    with path.open("wb") as f:
        jnp.savez(f, **cpu_params)


def restore_params(path: Path) -> dict[str, jnp.ndarray]:
    with path.open("rb") as f:
        data = jnp.load(f)
        return {k: data[k] for k in data.files}


def exercise_best_run(runs: list[dict]) -> dict:
    """Exercise: return run with minimum validation_loss."""
    raise NotImplementedError("Implement this in solutions/lesson_22_solution.py")
