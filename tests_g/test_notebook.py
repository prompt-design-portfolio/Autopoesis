"""The Colab notebook (§33, kept; A5's "files come to me at the pre-check stage").

`nbcheck` compiles every code cell, which catches the defect it was built for — a broken string
literal that JSON validity and notebook loading both miss. It does not catch a *renamed symbol*:
`from civitas_g.world.spec import clocks_of` compiles fine and fails at run time, on Colab, in
front of whoever is trying to reproduce a number. So these tests resolve every import the notebook
makes.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

import nbcheck

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "Civitas_G_Colab.ipynb"


def code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_the_notebook_exists_and_every_cell_compiles():
    assert NOTEBOOK.exists()
    assert nbcheck.check_file(NOTEBOOK) >= 5


def test_every_symbol_the_notebook_imports_from_civitas_g_exists():
    """The one nbcheck structurally cannot catch."""
    missing: list[str] = []
    for source in code_cells():
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith(("civitas_g", "nbcheck")):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if not hasattr(module, alias.name):
                    missing.append(f"{node.module}.{alias.name}")
    assert not missing, f"the notebook imports symbols that do not exist: {missing}"


def test_the_notebook_says_which_mode_can_meet_the_gate():
    """A shorter run is a different experiment. A notebook that printed a diff without saying so
    would be handing over a number that looks like a result and is not one."""
    text = "\n".join(code_cells())
    assert 'MODE = "quick"' in text
    assert "reference" in text and "3000" in text
    assert "is NOT the gate" in text


def test_the_notebook_marks_the_single_backend_clause_unmet():
    """Colab gives one backend; D12 needs two. The report must say so rather than pass on one."""
    text = "\n".join(code_cells())
    assert "One backend on Colab" in text
    assert "UNMET" in text or "unmet" in text


def test_the_notebook_runs_the_selftests_before_the_reproduction():
    """B§6: run before any result is read; any failure halts."""
    cells = code_cells()
    selftest = next(i for i, c in enumerate(cells) if "run_selftests" in c)
    reproduce = next(i for i, c in enumerate(cells) if "run_g1" in c)
    assert selftest < reproduce
    assert "halt_on_failure" in cells[selftest]


def test_the_notebook_refuses_to_continue_if_a_pin_no_longer_matches():
    text = "\n".join(code_cells())
    assert "nothing below is a gate" in text


@pytest.mark.parametrize("path", sorted(
    [str(p) for p in Path(__file__).resolve().parent.parent.glob("*.ipynb")]
    + [str(p) for p in Path(__file__).resolve().parent.parent.glob("notebooks/*.ipynb")]))
def test_nbcheck_is_green_on_every_notebook(path):
    """A4.1's clause, as a test rather than as a command someone remembers to run."""
    assert nbcheck.check_file(path) >= 0
