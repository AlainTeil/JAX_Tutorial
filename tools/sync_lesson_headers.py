"""Sync lesson module teaching headers from the curriculum registry."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from course.curriculum import LESSONS  # noqa: E402

HEADER_RE = re.compile(
    r"\n# Learning objectives:\n(?:# .+\n)+# Recap:\n(?:# .+\n)+\n",
    flags=re.MULTILINE,
)


def module_path(module: str) -> Path:
    return ROOT / (module.replace(".", "/") + ".py")


def render_header(lesson_number: int) -> str:
    lesson = LESSONS[lesson_number - 1]
    lines = ["", "# Learning objectives:"]
    lines.extend(f"# - {objective}" for objective in lesson.objectives)
    lines.append("# Mental model:")
    lines.append(f"# - {lesson.mental_model}")
    lines.append("# Common mistakes:")
    lines.extend(f"# - {mistake}" for mistake in lesson.common_mistakes)
    lines.append("# Recap:")
    lines.extend(f"# - {checkpoint}" for checkpoint in lesson.checkpoint)
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    for lesson in LESSONS:
        path = module_path(lesson.module)
        text = path.read_text(encoding="utf-8")
        replacement = render_header(lesson.number)
        updated, count = HEADER_RE.subn(replacement, text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not replace header in {path}")
        path.write_text(updated, encoding="utf-8")
        print(f"synced {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
