# JAX Deep Learning Tutorial

This repository is a CPU-first, 30-lesson, workbook-style JAX course for learners who are comfortable with Python and NumPy-shaped array thinking and are new to JAX. It pairs importable lesson modules, reference solutions, executed notebooks, and tests so students can learn interactively while instructors keep the material reproducible.

## What You Will Learn

By the end of the course, students should be able to:

- Write JAX programs with explicit state, immutable arrays, and pure functions.
- Use `grad`, `jit`, `vmap`, `lax.scan`, JAX PRNG keys, and pytrees fluently.
- Build and train small models with Flax and Optax.
- Diagnose numerical stability, tracing, control-flow, and shape issues.
- Understand what single-device demos can and cannot prove about `pmap`, `pjit`, and sharding.
- Save, restore, evaluate, compare, and report compact end-to-end models.
- Measure performance, prepare fixed-shape input pipelines, and package deployment-facing summaries.

## Requirements

- Python 3.12.
- CPU JAX by default.
- Optional accelerator or multi-device hardware for the advanced scaling extensions.

The dependency target is pinned around the validated local stack:

- JAX / jaxlib 0.9.x
- Flax 0.12.x
- Optax 0.2.x
- pytest 9.x

Use [constraints.txt](constraints.txt) for reproducible classroom installs.

## Quick Start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -c constraints.txt
python -m pytest
```

If `.venv/` already exists, `python3.12 -m venv .venv` reuses that directory and refreshes the virtual-environment metadata/scripts without deleting installed packages or other files. To rebuild the environment from scratch, remove `.venv/` first and then rerun the Quick Start commands.

For contributor tooling:

```bash
python -m pip install -e '.[dev]' -c constraints.txt
ruff format --check .
ruff check .
pyright
pytest
```

The editable install makes the `course`, `lessons`, and `solutions` packages importable while adding development tools such as Ruff and Pyright.

## Repository Layout

- [course/](course/) contains the curriculum registry used by notebooks, docs, and validation tests.
- [lessons/](lessons/) contains executable Python lesson modules and exercise stubs.
- [solutions/](solutions/) contains reference implementations for exercises.
- [notebooks/](notebooks/) contains executed workbook notebooks with TODO prompts, hints, and saved outputs.
- [tests/](tests/) validates lessons, solutions, notebooks, metadata, traceability, and numerical edge cases.
- [tools/](tools/) contains generation/synchronization scripts for notebooks and lesson headers.
- [docs/](docs/) contains contributor workflow notes.

## How To Study A Lesson

1. Open the matching notebook in [notebooks/](notebooks/).
2. Read the mental model before running the guided demo.
3. Run the code cells in order and compare your results with the saved outputs.
4. Work through the TODO exercise prompt in a scratch cell or in the corresponding lesson stub.
5. Check the matching reference implementation in [solutions/](solutions/) only after you have tried the exercise.
6. Run the lesson tests or the full test suite before moving on.

The notebooks are intentionally executable as published. Exercise cells name the TODO and reference solution while keeping CI green; students can replace those scratch cells locally while studying.

## Curriculum Map

| Phase | Lessons | Focus |
|---|---:|---|
| Foundations | 1-4 | JAX mental model, arrays, autodiff, and compilation |
| Transform Patterns | 5-8 | `vmap`, JAX control flow, randomness, and pytrees |
| Training Foundations | 9-12 | Linear models, optimization, minibatching, and stable losses |
| Advanced Single-Device JAX | 13-14 | Higher-order autodiff, debugging, profiling, and JAXPR inspection |
| Neural Network Stack | 15-18 | Flax modules, Optax training, evaluation, regularization, and sequence scans |
| Scaling and Systems | 19-21 | `pmap`, `pjit`, tensor sharding, and single-device fallback boundaries |
| Production Workflow | 22-24 | Checkpointing, ecosystem orientation, and capstone evaluation |
| Integration Labs | 25-30 | Transform composition, performance, data input, readiness checks, experiment analysis, and deployment reports |

The authoritative per-lesson metadata lives in [course/curriculum.py](course/curriculum.py). That registry lets the course evolve without relying on duplicated notebook prose.

## Recommended Study Paths

- Core path: lessons 1-18, then 24-25.
- Scaling path: lessons 1-12, then 19-21 and 28.
- Production path: lessons 1-12, then 15-18 and 22-30.
- Integration path: complete lessons 1-30 in order for the full course arc.
- Fast review path: run the notebooks first, then inspect lesson modules and tests for details.

## Advanced JAX Hardware Notes

Lessons 19-21 run on a single CPU device and label that behavior as a fallback. Single-device execution is useful for learning API shapes and placement metadata, but it is not evidence of speedup or real cross-device communication.

For optional multi-device CPU experiments, set the relevant XLA flag before importing JAX in a fresh process:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 python -m pytest -m multidevice
```

Do not set that flag midway through a notebook after JAX has already imported; restart the kernel first.

## Regenerating Course Artifacts

After editing [course/curriculum.py](course/curriculum.py), regenerate derived artifacts:

```bash
python tools/sync_lesson_headers.py
python tools/generate_notebooks.py
ruff format --check .
ruff check .
pyright
pytest
```

The notebook generator executes cells and stores outputs, so generation also works as a smoke test for the published workbook path.

## Troubleshooting

- Import errors from notebooks usually mean the notebook is not running from the repository root. Start Jupyter from this directory.
- CUDA warnings are expected on machines with an NVIDIA GPU but CPU-only `jaxlib`; the course falls back to CPU.
- Tracer errors usually mean Python control flow or side effects are being used where JAX needs staged array control flow.
- Shape errors are often more useful than they look. Check the lesson's checkpoint bullets and print the leading batch axes.
- If dependency resolution changes behavior, reinstall with `-c constraints.txt`.

## Validation Status

The refactored 30-lesson baseline is validated by Ruff, Pyright, pytest, notebook execution checks, curriculum metadata checks, and CI configuration in [.github/workflows/ci.yml](.github/workflows/ci.yml).
