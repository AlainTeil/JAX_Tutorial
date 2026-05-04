"""Lesson 24: End-to-end capstone project.

Build and train a compact MLP classifier using JAX + Flax + Optax with
functional training loops and deterministic toy data.
"""

# Learning objectives:
# - Train a compact Flax classifier end to end on deterministic toy data.
# - Save, restore, evaluate, and report model behavior with vectorized metrics.
# - Tie together PRNGs, pytrees, losses, Optax state, and evaluation checks.
# Mental model:
# - A JAX project is a set of explicit state transitions: data to params, params to loss, optimizer state to new state, and predictions to metrics.
# Common mistakes:
# - Ending with imperative metric code that cannot compose with JAX transformations.
# - Reporting accuracy without inspecting confusion matrix counts.
# Recap:
# - Training improves loss and reaches high accuracy on XOR.
# - Saved and restored parameters preserve the trained tree structure.
# - Confusion matrix counts sum to the number of evaluated examples.

import pickle
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class CapstoneMLP(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def make_xor_data() -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    y = jnp.array([0, 1, 1, 0], dtype=jnp.int32)
    return x, y


def create_state(seed: int = 0, learning_rate: float = 0.1):
    model = CapstoneMLP(hidden_dim=16, out_dim=2)
    key = jax.random.key(seed)
    params = model.init(key, jnp.zeros((1, 2), dtype=jnp.float32))["params"]
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model


def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def compute_loss(params: dict, apply_fn, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = apply_fn({"params": params}, x)
    return cross_entropy(logits, y)


@jax.jit
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    def loss_fn(p):
        return compute_loss(p, state.apply_fn, x, y)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def evaluate(
    state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    logits = state.apply_fn({"params": state.params}, x)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean((preds == y).astype(jnp.float32))
    loss = cross_entropy(logits, y)
    return {"loss": loss, "accuracy": acc}


def save_capstone_checkpoint(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(params, f)


def load_capstone_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        return pickle.load(f)


def evaluation_report(
    state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray
) -> dict[str, jnp.ndarray]:
    logits = state.apply_fn({"params": state.params}, x)
    preds = jnp.argmax(logits, axis=-1)
    acc = jnp.mean((preds == y).astype(jnp.float32))

    tp = jnp.sum((preds == 1) & (y == 1))
    fp = jnp.sum((preds == 1) & (y == 0))
    fn = jnp.sum((preds == 0) & (y == 1))
    precision = tp / jnp.maximum(tp + fp, 1)
    recall = tp / jnp.maximum(tp + fn, 1)
    f1 = 2.0 * precision * recall / jnp.maximum(precision + recall, 1e-8)

    conf = jnp.zeros((2, 2), dtype=jnp.int32).at[y, preds].add(1)

    return {
        "accuracy": acc,
        "precision": precision.astype(jnp.float32),
        "recall": recall.astype(jnp.float32),
        "f1": f1.astype(jnp.float32),
        "confusion_matrix": conf,
    }


def run_capstone_training(num_steps: int = 400, learning_rate: float = 0.1):
    x, y = make_xor_data()
    state, _ = create_state(seed=0, learning_rate=learning_rate)
    losses = []
    for _ in range(num_steps):
        state, loss = train_step(state, x, y)
        losses.append(loss)
    metrics = evaluate(state, x, y)
    return state, jnp.array(losses), metrics


def exercise_confusion_matrix(
    preds: jnp.ndarray, labels: jnp.ndarray, num_classes: int
) -> jnp.ndarray:
    """Exercise: build confusion matrix counts."""
    raise NotImplementedError("Implement this in solutions/lesson_24_solution.py")
