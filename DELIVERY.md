# Delivery step — run before any file leaves this repo

## Notebooks: compile every code cell, fail the delivery on any SyntaxError

```bash
python3 nbcheck.py                 # every *.ipynb in the repo
python3 nbcheck.py path/to.ipynb   # one
```

Exit code is non-zero if any code cell will not compile. **A notebook that fails this is not
delivered.** The generators (`make_notebook_v3_*.py`) call `nbcheck.write_checked(nb, path)`, which
compiles every cell *before* writing, so a broken notebook is never produced in the first place.

### The defect this exists to stop — it shipped three times

A code cell reached the user with a broken string literal:

```python
print("
in the checkpoint:", ...)
```

**Cause, in the writer.** The cell source was held in a **non-raw** triple-quoted literal,
`CODE_RUN = '''...'''`. A `\n` written inside that literal is a *real newline* by the time the
generator runs, so `print("\nfoo")` reaches the notebook as a string that opens and never closes.

It got there because these generators were authored through nested heredocs and string
replacements, and **every such round strips one backslash level**: `\\n` becomes `\n`, and once it
is `\n` in a non-raw literal it is a newline.

**Why nothing caught it.** The JSON was valid, the notebook loaded, the file was written, and the
generator exited 0. Validity of the container says nothing about the validity of the code inside it.

### Two defences, both required

1. **Raw literals.** Every `CODE_*` block in every generator is `r'''...'''`, so no escape can be
   eaten by a later edit. In a raw literal, write `\n` to emit the two-character escape.
2. **The compile gate.** `nbcheck.check_cells` compiles every code cell. Prevention can regress;
   the gate is what makes the regression visible before delivery rather than after.

## Everything else

- Python modules: `python3 -c "import ast; ast.parse(open('sim.py').read())"` after every edit.
- The sim's own self-tests before any result is read: `world_semantics_selftest`,
  `founder_tag_selftest`, `replay_mapping_selftest`, `learning_rule_selftest`, and
  `analysis.frozen_selftest`. The notebook setup cell runs all five and halts on failure.
