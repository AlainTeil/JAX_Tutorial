"""Lesson 16: Training loops with Flax + Optax."""

# Learning objectives:
# - Create a Flax `TrainState` with model params and an Optax optimizer.
# - Write a jitted train step that updates state functionally.
# - Track loss and accuracy for a tiny classifier.
# Mental model:
# - Flax owns the model definition, Optax owns optimizer transformations, and `TrainState` carries the explicit mutable-looking state as immutable values.
# Common mistakes:
# - Forgetting to pass `apply_fn` and params through the loss function explicitly.
# - Updating parameters without carrying optimizer state forward.
# Recap:
# - Training loss decreases on the toy dataset.
# - The state step increments as training progresses.
# - Accuracy is a scalar in the range [0, 1].

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state


class Classifier(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def make_toy_classification_data() -> tuple[jnp.ndarray, jnp.ndarray]:
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


def create_train_state(key: jax.Array, learning_rate: float = 0.1):
    model = Classifier(hidden_dim=8, out_dim=2)
    params = model.init(key, jnp.zeros((1, 2), dtype=jnp.float32))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx), model


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(labels.shape[0]), labels])


def compute_loss(params, apply_fn, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    logits = apply_fn({"params": params}, x)
    return cross_entropy_loss(logits, y)


@jax.jit
def train_step(state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray):
    grad_fn = jax.value_and_grad(compute_loss)
    loss, grads = grad_fn(state.params, state.apply_fn, x, y)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_for_steps(num_steps: int = 100, learning_rate: float = 0.1):
    key = jax.random.key(0)
    x, y = make_toy_classification_data()
    state, _ = create_train_state(key, learning_rate=learning_rate)
    losses = []
    for _ in range(num_steps):
        state, loss = train_step(state, x, y)
        losses.append(loss)
    return state, jnp.array(losses), x, y


def exercise_classification_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Exercise: compute classification accuracy from logits."""
    raise NotImplementedError("Implement this in solutions/lesson_16_solution.py")
