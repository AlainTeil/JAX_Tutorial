"""Lesson 29: Experiment analysis and comparison."""

# Learning objectives:
# - Rank experiment runs by validation loss and runtime.
# - Attach environment metadata to make comparisons reproducible.
# - Identify Pareto-efficient runs when accuracy and cost trade off.
# Mental model:
# - Experiment analysis turns run metadata into decisions. Keep metrics and costs explicit so ranking rules can be reviewed.
# Common mistakes:
# - Choosing the lowest loss without checking whether the runtime cost is acceptable.
# - Comparing runs without recording package or environment context.
# Recap:
# - The best run by validation loss is easy to identify from ranked metadata.
# - Environment fields are copied into the run summary without changing metric values.
# - Pareto selection keeps only runs that preserve a useful loss/runtime tradeoff.


def make_demo_runs() -> list[dict[str, float | int | str]]:
    return [
        {"run_name": "baseline", "seed": 0, "validation_loss": 0.42, "runtime_seconds": 8.0},
        {"run_name": "wider", "seed": 1, "validation_loss": 0.31, "runtime_seconds": 12.0},
        {"run_name": "fast", "seed": 2, "validation_loss": 0.36, "runtime_seconds": 6.0},
    ]


def rank_runs(runs: list[dict]) -> list[dict]:
    return sorted(runs, key=lambda run: (run["validation_loss"], run["runtime_seconds"]))


def best_run(runs: list[dict]) -> dict:
    return rank_runs(runs)[0]


def loss_delta(run: dict, baseline: dict) -> float:
    return float(run["validation_loss"] - baseline["validation_loss"])


def attach_environment(metadata: dict, package_versions: dict[str, str]) -> dict:
    return {**metadata, "packages": dict(sorted(package_versions.items()))}


def summarize_runs(runs: list[dict]) -> dict[str, object]:
    ranked = rank_runs(runs)
    return {
        "num_runs": len(runs),
        "best_run_name": ranked[0]["run_name"],
        "mean_validation_loss": sum(float(run["validation_loss"]) for run in runs) / len(runs),
    }


def exercise_select_pareto_runs(runs: list[dict]) -> list[dict]:
    """Exercise: select runs not dominated on validation loss and runtime."""
    raise NotImplementedError("Implement this in solutions/lesson_29_solution.py")
