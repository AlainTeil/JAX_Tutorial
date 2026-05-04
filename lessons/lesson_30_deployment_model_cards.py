"""Lesson 30: Deployment checks and model cards."""

# Learning objectives:
# - Convert model scores into reviewable prediction records.
# - Summarize calibration bins for simple deployment checks.
# - Package metrics and limitations into a lightweight model card.
# Mental model:
# - Deployment is another transformation boundary: numeric outputs become decisions, reports, and operational constraints.
# Common mistakes:
# - Shipping a threshold without documenting how it changes precision and recall.
# - Reporting metrics without limitations or intended-use context.
# Recap:
# - Prediction tables preserve IDs, scores, and thresholded decisions.
# - Calibration bins report counts, confidence, and accuracy per range.
# - Model cards contain metrics and limitations in serializable fields.

import jax
import jax.numpy as jnp


def sigmoid_scores(logits: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.sigmoid(logits)


def threshold_predictions(probs: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    return (probs >= threshold).astype(jnp.int32)


def prediction_table(
    example_ids: list[str], probs: jnp.ndarray, threshold: float = 0.5
) -> list[dict]:
    preds = threshold_predictions(probs, threshold)
    return [
        {"id": example_id, "score": float(score), "prediction": int(pred)}
        for example_id, score, pred in zip(example_ids, probs, preds, strict=True)
    ]


def calibration_bins(
    probs: jnp.ndarray, labels: jnp.ndarray, num_bins: int = 2
) -> list[dict[str, float | int]]:
    edges = jnp.linspace(0.0, 1.0, num_bins + 1)
    rows = []
    for idx in range(num_bins):
        lower, upper = edges[idx], edges[idx + 1]
        mask = (probs >= lower) & (probs <= upper if idx == num_bins - 1 else probs < upper)
        count = int(jnp.sum(mask))
        if count == 0:
            rows.append({"bin": idx, "count": 0, "confidence": 0.0, "accuracy": 0.0})
            continue
        bin_probs = probs[mask]
        bin_labels = labels[mask]
        bin_preds = threshold_predictions(bin_probs)
        rows.append(
            {
                "bin": idx,
                "count": count,
                "confidence": float(jnp.mean(bin_probs)),
                "accuracy": float(jnp.mean((bin_preds == bin_labels).astype(jnp.float32))),
            }
        )
    return rows


def model_card(summary: str, metrics: dict[str, float], limitations: list[str]) -> dict:
    return {"summary": summary, "metrics": dict(metrics), "limitations": list(limitations)}


def exercise_apply_threshold(probs: jnp.ndarray, threshold: float) -> jnp.ndarray:
    """Exercise: convert probabilities into integer class predictions."""
    raise NotImplementedError("Implement this in solutions/lesson_30_solution.py")
