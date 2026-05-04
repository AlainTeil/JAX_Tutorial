from __future__ import annotations

import json
from pathlib import Path

from course.curriculum import LESSONS

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def test_notebooks_have_kernel_metadata_outputs_and_exercises():
    for lesson in LESSONS:
        nb = _load(ROOT / lesson.notebook)
        assert nb.get("metadata", {}).get("kernelspec", {}).get("name") == "python3"
        code_cells = [cell for cell in nb.get("cells", []) if cell.get("cell_type") == "code"]
        assert len(code_cells) >= 3
        assert any(cell.get("outputs") for cell in code_cells), lesson.notebook

        full_text = "\n".join(_source(cell) for cell in nb.get("cells", []))
        assert lesson.exercise_name in full_text
        assert "TODO:" in full_text
        assert "Expected Output Checkpoint" in full_text
        assert "Loss-like values should decrease where training is present" not in full_text


def test_checkpoint_text_is_lesson_specific():
    checkpoints = []
    for lesson in LESSONS:
        nb = _load(ROOT / lesson.notebook)
        checkpoint_cells = [
            _source(cell)
            for cell in nb.get("cells", [])
            if cell.get("cell_type") == "markdown" and "Expected Output Checkpoint" in _source(cell)
        ]
        assert len(checkpoint_cells) == 1
        checkpoints.append(checkpoint_cells[0])
    assert len(set(checkpoints)) == len(checkpoints)
