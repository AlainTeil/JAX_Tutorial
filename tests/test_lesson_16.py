from lessons.lesson_16_flax_optax_training import (
    compute_loss,
    train_for_steps,
)
from solutions.lesson_16_solution import classification_accuracy


def test_training_reduces_loss():
    state, losses, x, y = train_for_steps(num_steps=80, learning_rate=0.1)
    assert float(losses[-1]) < float(losses[0])

    final_loss = compute_loss(state.params, state.apply_fn, x, y)
    assert final_loss < losses[0]


def test_solution_accuracy_in_range():
    state, _, x, y = train_for_steps(num_steps=30, learning_rate=0.1)
    logits = state.apply_fn({"params": state.params}, x)
    acc = classification_accuracy(logits, y)
    assert 0.0 <= float(acc) <= 1.0
