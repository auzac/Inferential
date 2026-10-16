# CLAUDE.md

Inferential Model Studio — offline PySide6 desktop app for industrial inferential (soft-sensor)
models. Windows-targeted; runs on macOS through the local `.venv`.

`STRUCTURE.md` maps every file and says where to look for a given change; `docs/methodology.md`
fixes the maths, time and alignment conventions. Read those rather than restating them here.

## Standing instructions

- **Work style: use the `ponytail` skill** (`.claude/skills/ponytail/SKILL.md`) — the smallest
  change that fully solves the task.
- **`core/` never imports Qt** (it must run in Jupyter / Seeq Data Lab); `ui/` pages display state
  and call `core/`, implementing no algorithms. `tests/unit/test_core_isolation.py` enforces this.
- **A new project setting** goes in `core/schemas.py`, then bump `SCHEMA_VERSION` and add the
  migration in `core/persistence.py`.
- **Before calling it done:** `ruff check src tests` and `pytest`.

## Run

```bash
.venv/bin/python -m inferential_studio               # GUI
.venv/bin/python -m inferential_studio --smoke-test  # build the window, exit 0
```
