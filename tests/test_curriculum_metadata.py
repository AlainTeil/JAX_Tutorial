from __future__ import annotations

import importlib
from pathlib import Path

from course.curriculum import LESSONS, lesson_numbers

ROOT = Path(__file__).resolve().parent.parent


def test_curriculum_registry_is_ordered_and_unique():
    numbers = lesson_numbers()
    assert numbers == sorted(numbers)
    assert len(numbers) == len(set(numbers))
    assert len(LESSONS) >= 30


def test_curriculum_files_and_exercises_exist():
    for lesson in LESSONS:
        module = importlib.import_module(lesson.module)
        solution = importlib.import_module(lesson.solution)
        assert hasattr(module, lesson.exercise_name)
        assert hasattr(solution, lesson.solution_function)
        assert (ROOT / lesson.notebook).exists()


def test_curriculum_text_is_lesson_specific():
    generic_fragments = (
        "Understand and apply the core ideas",
        "Implement the main transformation or training pattern safely",
        "this lesson's core concept",
    )
    for lesson in LESSONS:
        combined = "\n".join(
            [lesson.title, lesson.mental_model, lesson.exercise_prompt, *lesson.objectives]
        )
        assert all(fragment not in combined for fragment in generic_fragments)
        assert len(set(lesson.objectives)) == len(lesson.objectives)


def test_lesson_modules_do_not_use_generic_teaching_headers():
    generic_fragments = (
        "Understand and apply the core ideas",
        "Implement the main transformation or training pattern safely",
        "this lesson's core concept",
    )
    for lesson in LESSONS:
        module_path = ROOT / (lesson.module.replace(".", "/") + ".py")
        text = module_path.read_text(encoding="utf-8")
        assert all(fragment not in text for fragment in generic_fragments)
        assert lesson.mental_model in text
