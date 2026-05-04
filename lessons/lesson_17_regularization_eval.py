"""Lesson 17: Regularization and model evaluation."""

# Learning objectives:
# - Add L2 regularization to a binary classifier loss.
# - Compute accuracy, precision, and recall from logits.
# - Interpret F1 as a balance between precision and recall.
# Mental model:
# - Evaluation metrics are array programs too. Make thresholds and reductions explicit so they can be tested and transformed.
# Common mistakes:
# - Reporting accuracy alone on imbalanced classification examples.
# - Dividing by zero when no positive predictions or labels are present.
# Recap:
# - Regularized loss is at least as large as the data loss when weight decay is positive.
# - Precision and recall are bounded scalar metrics.
# - F1 is zero-safe and bounded between 0 and 1.

import jax
import jax.numpy as jnp


def linear_logits(params: dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    return (x @ params["w"] + params["b"]).squeeze(-1)


def bce_loss_from_logits(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(
        jnp.maximum(logits, 0.0) - logits * targets + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )


def l2_penalty(params: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return jnp.sum(params["w"] ** 2)


def total_loss(
    params: dict[str, jnp.ndarray],
    x: jnp.ndarray,
    y: jnp.ndarray,
    weight_decay: float,
) -> jnp.ndarray:
    logits = linear_logits(params, x)
    return bce_loss_from_logits(logits, y) + weight_decay * l2_penalty(params)


def accuracy_from_logits(logits: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
    return jnp.mean((preds == y).astype(jnp.float32))


def precision_recall_from_logits(
    logits: jnp.ndarray, y: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    preds = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.float32)
    tp = jnp.sum((preds == 1.0) & (y == 1.0))
    fp = jnp.sum((preds == 1.0) & (y == 0.0))
    fn = jnp.sum((preds == 0.0) & (y == 1.0))
    precision = tp / jnp.maximum(tp + fp, 1.0)
    recall = tp / jnp.maximum(tp + fn, 1.0)
    return precision.astype(jnp.float32), recall.astype(jnp.float32)


def exercise_f1_score(precision: jnp.ndarray, recall: jnp.ndarray) -> jnp.ndarray:
    """Exercise: compute F1 score from precision and recall."""
    raise NotImplementedError("Implement this in solutions/lesson_17_solution.py")
