# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a statistical methods reproduction repository (Python 3 primary). See `CLAUDE.md` for project structure and workflow details.

### Environment

- Python virtual environment lives at `/workspace/.venv`. Activate with `source /workspace/.venv/bin/activate`.
- Core dev tools: `pytest` (testing), `ruff` (linting), `numpy`/`scipy` (numerical).
- `uv` is installed at `~/.local/bin/uv` — ensure `$HOME/.local/bin` is on `PATH`.

### Key commands

- **Lint:** `ruff check main/`
- **Test:** `pytest main/` (or target a specific sub-project, e.g. `pytest main/bootstrap_ci/tests/`)
- **Run a sub-project:** `python main/<project>/src/<script>.py`

### Gotchas

- System site-packages are not writable; always use the venv or `uv pip install` into the venv.
- Each sub-project's `tests/` adds `src/` to `sys.path` locally — no package install step is needed to run tests.
