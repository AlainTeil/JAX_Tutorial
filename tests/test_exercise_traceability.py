import re
from pathlib import Path

LESSONS_DIR = Path(__file__).resolve().parent.parent / "lessons"
SOLUTIONS_DIR = Path(__file__).resolve().parent.parent / "solutions"


def _exercise_functions(text: str) -> list[str]:
    return re.findall(r"def\s+(exercise_[a-zA-Z0-9_]+)\s*\(", text)


def test_each_lesson_has_exercise_stub_and_solution_file():
    for lesson_path in sorted(LESSONS_DIR.glob("lesson_*.py")):
        text = lesson_path.read_text(encoding="utf-8")
        exercises = _exercise_functions(text)
        assert exercises, f"No exercise function in {lesson_path.name}"

        suffix = lesson_path.stem.split("_", 2)[1]
        solution_path = SOLUTIONS_DIR / f"lesson_{suffix}_solution.py"
        assert solution_path.exists(), f"Missing solution file for {lesson_path.name}"


def test_exercise_stub_references_solution_file_in_error_message():
    for lesson_path in sorted(LESSONS_DIR.glob("lesson_*.py")):
        text = lesson_path.read_text(encoding="utf-8")
        suffix = lesson_path.stem.split("_", 2)[1]
        assert f"solutions/lesson_{suffix}_solution.py" in text
