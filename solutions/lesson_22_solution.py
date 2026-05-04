"""Solutions for Lesson 22 exercises."""


def best_run(runs: list[dict]) -> dict:
    return min(runs, key=lambda r: r["validation_loss"])
