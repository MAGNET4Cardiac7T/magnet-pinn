# Using uv (instead of Poetry)

This project uses [uv](https://docs.astral.sh/uv/) as its package and project manager.

## Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip / pipx
pip install uv
```

## Common commands

| Poetry command | uv equivalent | Description |
|---|---|---|
| `poetry install` | `uv sync` | Install all dependencies (including dev) |
| `poetry install --no-dev` | `uv sync --no-dev` | Install without dev dependencies |
| `poetry add <pkg>` | `uv add <pkg>` | Add a runtime dependency |
| `poetry add --group dev <pkg>` | `uv add --group dev <pkg>` | Add a dev dependency |
| `poetry remove <pkg>` | `uv remove <pkg>` | Remove a dependency |
| `poetry run <cmd>` | `uv run <cmd>` | Run a command in the project virtualenv |
| `poetry build` | `uv build` | Build sdist and wheel |
| `poetry publish` | `uv publish` | Publish to PyPI |
| `poetry lock` | `uv lock` | Regenerate the lockfile |
| `poetry update` | `uv lock --upgrade` | Upgrade all dependencies |
| `poetry update <pkg>` | `uv lock --upgrade-package <pkg>` | Upgrade a single dependency |
| `poetry shell` | `source .venv/bin/activate` | Activate the virtualenv (uv has no shell wrapper) |

## Day-to-day workflow

```bash
# Clone and set up
git clone <repo-url> && cd magnet-pinn
uv sync            # creates .venv and installs everything

# Run tests
uv run pytest

# Run a script
uv run python examples/some_example.py

# Add a new dependency
uv add requests                   # runtime
uv add --group dev ruff           # dev-only

# Build the package
uv build          # outputs to dist/

# Publish to PyPI (set UV_PUBLISH_TOKEN or pass --token)
uv publish
```

## Notes

- uv automatically creates a `.venv` in the project root.
- The lockfile (`uv.lock`) is not committed for this library (see `.gitignore`).
- Pre-commit hooks already use `uv run` — no extra setup needed.
