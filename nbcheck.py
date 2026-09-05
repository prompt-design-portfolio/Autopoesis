"""Delivery gate for generated notebooks.

THE DEFECT THIS EXISTS TO STOP, which shipped three times:
a code cell whose source contains a broken string literal, because the generator held that cell in
a NON-RAW triple-quoted literal.  `\\n` written inside `'''...'''` is a real newline by the time
the generator runs, so `print("\\nfoo")` reaches the notebook as

    print("
    foo")

which is a SyntaxError the moment the cell is executed -- and nothing in JSON validity, notebook
loading, or "it wrote the file" catches it.  Every earlier check I had passed on those files.

Two defences, and both are needed:
  1. the generators hold cell source in RAW literals (r'''...'''), so no escape can be eaten;
  2. this module compiles every code cell before the notebook is allowed to be written.
"""
import json
import pathlib


class NotebookCheckFailed(Exception):
    pass


def check_cells(cells, name="<notebook>"):
    """compile() every code cell.  Raises NotebookCheckFailed listing every bad cell."""
    bad = []
    for i, c in enumerate(cells):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        try:
            compile(src, f"{name}:cell{i}", "exec")
        except SyntaxError as e:
            lines = src.splitlines()
            lo = max(0, (e.lineno or 1) - 3)
            ctx = "\n".join(f"      {n:3d}| {l}" for n, l in enumerate(lines[lo:(e.lineno or 1)],
                                                                      start=lo + 1))
            bad.append(f"  cell {i}: {type(e).__name__} at line {e.lineno}: {e.msg}\n{ctx}")
    if bad:
        raise NotebookCheckFailed(
            f"{name}: {len(bad)} code cell(s) will not compile -- NOT delivering.\n"
            + "\n".join(bad))
    return len([c for c in cells if c.get("cell_type") == "code"])


def check_file(path):
    path = pathlib.Path(path)
    nb = json.loads(path.read_text())
    return check_cells(nb["cells"], path.name)


def write_checked(nb, path):
    """Compile every code cell, THEN write.  A notebook that would not run is never written."""
    n = check_cells(nb["cells"], pathlib.Path(path).name)
    pathlib.Path(path).write_text(json.dumps(nb, indent=1))
    print(f"wrote {path}  ({n} code cells, all compile)")
    return path


if __name__ == "__main__":
    import sys
    targets = sys.argv[1:] or sorted(str(p) for p in pathlib.Path(".").glob("*.ipynb"))
    fails = 0
    for t in targets:
        try:
            n = check_file(t)
            print(f"PASS  {t}  ({n} code cells)")
        except NotebookCheckFailed as e:
            fails += 1
            print(f"FAIL  {e}")
    print(f"\n{len(targets) - fails}/{len(targets)} notebooks pass")
    sys.exit(1 if fails else 0)
