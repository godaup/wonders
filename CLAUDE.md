# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`wonders` is a Python implementation of a "7 Wonders"-style card game, intended to be **optimized for Deep Learning integration** (i.e., the game engine should expose clean, machine-consumable state/action interfaces rather than being human-UI-driven). The project is in early Pre-Alpha: the domain model is being scaffolded and most classes are stubs.

## Tooling & Commands

Dependencies are managed with a plain **virtual environment + pip** (PEP 621 metadata in `pyproject.toml`, setuptools build backend). Set up for local dev:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test,doc,dev]"
```

Optional-dependency groups: `test`, `doc`, `dev` (defined under `[project.optional-dependencies]`).

The `makefile` is the primary task runner (activate the venv first):

- `make test` — runs `format`, `lint`, then `unittest` (the full local gate)
- `make format` — `isort` + `black` (line length **120**)
- `make lint` — `flake8` + `mypy` (both must pass; mypy is enforced)
- `make unittest` — `pytest`
- `make coverage` — pytest with branch coverage report

Run a single test:

```
pytest tests/test_wonders.py::test_command_line_interface
```

Multi-version testing (py36–py39) is done via `tox`, which mirrors CI.

The console entry point `wonders` maps to `wonders.cli:main` (Click-based).

## CI / Release

- CI (`.github/workflows/dev.yml`, `preview.yml`) runs the tox matrix across Python 3.6–3.9 plus format/lint/build stages defined in `setup.cfg`.
- Releasing is tag-driven: bump with `bump2version {major|minor|patch}` (updates `pyproject.toml` and `wonders/__init__.py`, commits, and tags per `.bumpversion.cfg`), then `git push --tags`. `release.yml` builds with `python -m build` and deploys to PyPI on passing tests. Add a `CHANGELOG.md` entry before bumping.

## Architecture

The game model is split into three domains under `wonders/`, each a package with a `base.py` defining core abstractions:

- **`wonders/cards/`** — `base.py` holds the `Card` class plus the `Color` and `Ressource` enums (note the spelling: `Ressource`). A `Card` carries a `color`, a list of `costs` (`Ressource`s), chaining links (`chain_requires` / `chain_enables`, referencing other card names), and a list of `Effect`s. This is the most developed part of the model.
- **`wonders/effects/`** — `Effect` (base.py) is the extension point for what a card *does* when played. Currently an empty base class; concrete effects are expected to subclass it. `Card.effects` is typed against this base.
- **`wonders/engine/`** — game-flow orchestration. `Phase` (phase.py) is a stub for the turn/round phase state machine.

`wonders/wonders.py` is the legacy cookiecutter "main module" placeholder and is effectively unused — real logic lives in the subpackages above.

Key design intent when extending: cards compose behavior via `Effect` objects rather than subclassing `Card`, and card interactions (chaining, resource costs) are data on the `Card` rather than hardcoded logic. Keep the engine's state representation explicit and serializable to serve the Deep Learning goal.

## Conventions

- `black` and `isort` are authoritative for formatting (line length 120, `skip-string-normalization`). Google-style docstrings; `flake8` ignores `E203, E266, W503`.
- Type annotations are expected — `mypy` runs in the lint gate (over `wonders` and `tests`).
- Pre-commit hooks (`.pre-commit-config.yaml`) enforce the above plus CRLF/whitespace fixes; run `make pre-commit` to check all files.
