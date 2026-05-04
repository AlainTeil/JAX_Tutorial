"""Solutions for Lesson 29 exercises."""


def select_pareto_runs(runs: list[dict]) -> list[dict]:
    selected = []
    for run in runs:
        dominated = any(
            other is not run
            and other["validation_loss"] <= run["validation_loss"]
            and other["runtime_seconds"] <= run["runtime_seconds"]
            and (
                other["validation_loss"] < run["validation_loss"]
                or other["runtime_seconds"] < run["runtime_seconds"]
            )
            for other in runs
        )
        if not dominated:
            selected.append(run)
    return sorted(selected, key=lambda item: (item["validation_loss"], item["runtime_seconds"]))
