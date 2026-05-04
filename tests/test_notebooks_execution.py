import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def _load_notebook(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_all_notebooks_have_expected_output_checkpoint():
    for nb_path in sorted(NOTEBOOK_DIR.glob("lesson_*.ipynb")):
        nb = _load_notebook(nb_path)
        cells = nb.get("cells", [])
        assert any(
            c.get("cell_type") == "markdown"
            and any("Expected Output Checkpoint" in s for s in c.get("source", []))
            for c in cells
        ), f"Missing checkpoint cell in {nb_path.name}"


def test_execute_all_notebook_code_cells():
    for nb_path in sorted(NOTEBOOK_DIR.glob("lesson_*.ipynb")):
        nb = _load_notebook(nb_path)
        env = {"__name__": "__main__"}
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = cell.get("source", [])
            code = "\n".join(src) if isinstance(src, list) else str(src)
            code = code.replace("\\n", "\n")
            if code.strip():
                exec(compile(code, str(nb_path), "exec"), env, env)
