"""Solutions for Lesson 10 exercises."""


def adaptive_lr(step: int, base_lr: float) -> float:
    return base_lr / (1.0 + 0.1 * step)
