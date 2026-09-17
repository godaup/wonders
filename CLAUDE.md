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

The game implements the **7 Wonders** base game (3 Ages of 6 plays, draft-and-pass,
build/wonder/discard, neighbour trading, military conflict, and the full seven
scoring categories). It is split into three packages under `wonders/`:

- **`wonders/cards/`** — static content. `base.py` has the `Color` and `Ressource`
  enums (note the spelling; `GOLD` models coins and is *not* a tradeable resource),
  `ScienceSymbol`, the `Cost` value object (bank coins + a `Counter` of resources),
  and the `Card` dataclass (color, cost, `effects`, chaining, plus deck-composition
  metadata: `expansion`, `at_players`, `guild`). `wonder.py` has `Wonder`/`WonderStage`.
  **Content is data, not code**: cards and wonders live declaratively in
  `data/<expansion>.json` and are parsed by `content.py`, which maps a small typed
  *effect spec* (`{"type": "production", ...}`) to `Effect` objects via
  `EFFECT_BUILDERS`. `content.build_age_deck(content, age, n_players, rng)` assembles
  a deck honouring per-card copy counts (`at_players` thresholds ≤ N) and drawing
  `n_players + 2` guilds into Age III. To add a card, edit JSON; to add a *mechanic*,
  add an effect + register it. `library.py` is now thin loader-backed accessors for
  tests/demos. The shipped `data/base.json` is a *representative slice* with
  provisional `at_players` — the full deck + expansions (WHITE/BLACK) are the main
  outstanding data-entry. JSON is bundled via `[tool.setuptools.package-data]` and
  read with `importlib.resources`.
- **`wonders/effects/`** — what a card or wonder stage *does*. `base.py` defines the
  hook-based `Effect` (every effect answers the same uniform hooks —
  `production`, `shields`, `science`, `trade_rules`, `coins_on_build`,
  `victory_points` — with neutral defaults), plus `Context` (a player + its two
  neighbours) and `TradeRule`. `library.py` has the concrete effects; `Tally` is a
  flexible counter that expresses most yellow commercial cards and purple guilds.
- **`wonders/engine/`** — the rules. `player.py` (`Player` aggregates production /
  shields / science / trade-prices from all built effects), `trade.py`
  (`cheapest_payment`: the min-coin resource-payment solver — the core piece,
  own production free + neighbours priced, each slot used once per build),
  `game.py` (`Game`: payment, the three actions, legality, military resolution,
  scoring, the `play`/`play_age` draft loop — `play_age` takes an optional `on_turn`
  observer for recording), `actions.py`, `phase.py`, plus the DL/interface layer:
  `observation.py` (`build_observation` → a per-seat `Observation`: full public
  tableaus, own hand, legal actions, and a **history filtered for partial
  observability** — opponents' buried/discarded card identities are hidden) and
  `record.py` (`GameRecord`: a chess-PGN-style replayable notation — setup + ordered
  decks + every decision; `replay()` re-simulates deterministically without any RNG).
- **`wonders/agents/`** — the agent layer over the engine. `base.Agent` is the shared
  `decide(obs) -> Action` interface. `baselines.py` has `RandomAgent`, `GreedyAgent`,
  `DiscardAgent`, and colour-focused `ArchetypeAgent`s (`MilitaristAgent`,
  `ScientistAgent`, `CivicAgent`, `EconomistAgent`, `BuilderAgent`) — cheap, diverse
  opponents for self-play. `heuristic.HeuristicAgent` is the stronger baseline: a
  tunable value function over the Observation (VP, *marginal* science-set gain,
  Age-scaled military, economy, chaining tempo, wonder progression) that builds-first
  and is competitive with / beats greedy. `strategies.py` has characterful policies:
  `OpportunistAgent` (switches colour focus every few turns), `BlockerAgent`
  (denies the downstream neighbour their most-wanted card — Ages I/III pass left,
  Age II right), `TableAgent` (best affordable by a fixed value lookup table), and
  `BalancedAgent` (fills the scarcest colour). All read only the `Observation`, so
  they respect partial observability. `encoders.py` maps an `Observation` to a
  representation (`DictEncoder` structured, `FlatEncoder` fixed-length vector with
  optional history); `match.play_match` runs agents and returns `(Game, GameRecord)`.

Design rules to preserve when extending:
- **Cards compose behaviour via `Effect` objects**, never by subclassing `Card`;
  the engine introspects effects only through the uniform `Effect` hooks and never
  type-switches on concrete effect classes. Add a new mechanic by adding an effect.
- **Import direction is one-way**: `cards.base` imports nothing from `effects` at
  runtime (the `Effect` annotation is `TYPE_CHECKING`-only). Keep it that way — the
  cycle is what the scaffold originally tripped on.
- **DL-facing surface**: agents consume an `Observation` (via `build_observation`)
  and return a legal `Action`; `encoders` turn observations into tensors/dicts;
  `GameRecord` is the canonical serialized game. `Game.legal_actions`/`observe`/`score`
  remain the low-level primitives. Keep state explicit, serializable, and keep the
  Observation partial-observability-correct (never leak hidden cards to opponents).

`wonders/wonders.py` is now a thin public-API re-export (`from wonders.wonders import
Game, Player, ...`).

**Expansions are optional and combinable** (`wonders/ruleset.py`). A `Ruleset.of('leaders',
'cities')` composes any set of expansion ids (`base` always implied); combining is a
union. Content merges via `load_content('base', 'leaders', 'cities')` — each expansion is
one `data/<id>.json` file. Rule tweaks are designed to be inert when the expansion is
absent, so no special-casing: Leaders only raises the starting treasury (`Ruleset.starting_coins`,
threaded through the `GameRecord`); Cities adds `Player.debt`/`Player.diplomacy` (both default
0) consumed by scoring (debt = −1 VP) and military resolution (a diplomacy token removes a
player from the conflict ring). Cities' new effects (`Debt`, `Diplomacy`) use the generic
`Effect.on_build(ctx)` mutation hook. Leader (white) cards are drafted separately so
`build_age_deck` excludes them (`Content.leaders()`); Cities (black) cards carry `at_players`
and are dealt into the Ages. `armada`/`babel` are registered but flagged `implemented=False`
(`Ruleset.unimplemented`) — their subsystems (naval, tower) are not mechanized. The shipped
`leaders.json`/`cities.json` are **representative slices with original placeholder card names**
(not the official cards); accurate full sets are follow-up data entry.

## Conventions

- `black` and `isort` are authoritative for formatting (line length 120, `skip-string-normalization`). Google-style docstrings; `flake8` ignores `E203, E266, W503`.
- Type annotations are expected — `mypy` runs in the lint gate (over `wonders` and `tests`).
- The engine uses `dataclasses` and `from __future__ import annotations`, so it needs **Python 3.7+** in practice even though `pyproject.toml` classifiers still list 3.6.
- Pre-commit hooks (`.pre-commit-config.yaml`) enforce the above plus CRLF/whitespace fixes; run `make pre-commit` to check all files.
