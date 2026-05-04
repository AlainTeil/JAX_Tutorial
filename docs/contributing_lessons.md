# Contributing Lesson Material

Use [course/curriculum.py](../course/curriculum.py) as the source of truth for lesson-facing prose. The current course has 30 lessons, and notebooks plus lesson headers are generated from that registry so objectives, exercises, hints, checkpoints, and common mistakes stay aligned.

## Edit Workflow

1. Install contributor tooling from the repository root:

```bash
python -m pip install -e '.[dev]' -c constraints.txt
```

2. Update or add the lesson module in [lessons/](../lessons/).
3. Update the matching solution in [solutions/](../solutions/).
4. Update the lesson entry in [course/curriculum.py](../course/curriculum.py).
5. Add or update the generated demo snippet in [tools/generate_notebooks.py](../tools/generate_notebooks.py). Every `LessonSpec.number` needs a matching `DEMO_CODE` entry.
6. Run:

```bash
python tools/sync_lesson_headers.py
python tools/generate_notebooks.py
ruff format --check .
ruff check .
pyright
pytest
```

## Notebook Rules

- Every notebook should include a lesson-specific mental model, guided demo, TODO exercise prompt, hint, checkpoint, common mistakes, and extension.
- Every notebook should execute from the repository root without modifying tracked artifacts.
- Saved outputs are intentional. They help students compare a fresh run with the published result.
- Keep TODO cells executable by default. Students can replace them locally while working.

## Test Rules

- Lesson behavior belongs in focused `tests/test_lesson_*.py` files.
- Cross-artifact guarantees belong in traceability or curriculum metadata tests.
- Hardware-dependent behavior should be marked with `multidevice` and skipped cleanly when only one local device is available.
- Notebook changes should pass both execution tests and quality tests.

## Style Rules

- Prefer small pure functions with explicit arrays, keys, and state.
- Prefer vectorized JAX array updates over Python loops in final/reference solutions.
- Label single-device fallbacks clearly in distributed lessons.
- Keep examples deterministic and tiny enough for CPU-only CI.