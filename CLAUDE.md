# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`wonders` is a Python implementation of a "7 Wonders"-style card game, intended to be **optimized for Deep Learning integration** (i.e., the game engine should expose clean, machine-consumable state/action interfaces rather than being human-UI-driven). The base game is fully playable (all cards, wonders and wonder powers), with agents, a replayable game notation, training environments and a benchmark; expansions are partially mechanized. PyPI metadata still says Pre-Alpha.

## Tooling & Commands

Dependencies are managed with a plain **virtual environment + pip** (PEP 621 metadata in `pyproject.toml`, setuptools build backend). Set up for local dev:

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[test,dev]"
```

Optional-dependency groups: `test`, `dev`, `ml` (defined under `[project.optional-dependencies]`). There is no
docs site: the README is the entry point; design notes live in `docs/` and training reports in `sweeps/*.md`.

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

Multi-version testing (py311–py314) is done via `tox`, which mirrors CI.

The console entry point `wonders` maps to `wonders.cli:main`, a Click group: `wonders play` (one game, transcript + scores, `--record` saves JSON), `wonders replay <record.json>`, `wonders bench` (win rates with CIs, `-p` processes), `wonders agents`, `wonders gallery`, `wonders table`, `wonders gui` (local browser play), `wonders ext` (extensions and contribution checks). `wonders-gallery` / `wonders-table` remain as aliases.

## CI / Release

- CI (`.github/workflows/dev.yml`, `pre-commit.yml`) runs the tox matrix across Python 3.11–3.14 on Linux/macOS/Windows plus format/lint/build stages defined in `setup.cfg`. CI installs the package (not editable), so a file git does not track fails there — check `git status` for new files.
- Releasing is tag-driven: bump with `bump2version {major|minor|patch}` (updates `pyproject.toml` and `wonders/__init__.py`, commits, and tags per `.bumpversion.cfg`), then `git push --tags`. `release.yml` builds with `python -m build` and deploys to PyPI on passing tests. Add a `CHANGELOG.md` entry before bumping.

## Architecture

The game implements the **7 Wonders** base game (3 Ages of 6 plays, draft-and-pass,
build/wonder/discard, neighbour trading, military conflict, and the full seven
scoring categories). It is split into three packages under `wonders/`:

- **`wonders/cards/`** — static content. `base.py` has the `Color` and `Resource`
  enums (`GOLD` models coins and is *not* a tradeable resource),
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
  tests/demos. `data/base.json` is the complete base deck (68 cards,
  10 guilds, 7 wonders with A/B sides); the base deck supports 3–7 players. JSON is bundled via `[tool.setuptools.package-data]` and
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
  `game.py` (`Game`: payment, the actions, legality, military resolution, scoring,
  and the draft loop. **`Game.age_steps(age, hands)` is the single source of truth for
  turn order**: a coroutine that yields `DecisionPoint`s (`PLAY` for every seat, plus
  single-seat `LAST_CARD` / `DISCARD_BUILD` granted by wonder powers) and receives
  `{seat: Action}`; `play_age` (callback policies + `on_turn`/`on_extra` observers),
  `GameRecord.replay` and the env all drive it via `drive()`. `Game.decision` tells
  `legal_actions` which kind of decision is open), `actions.py` (`ActionKind`
  BUILD/WONDER/DISCARD/FREE, `DecisionKind`, `DecisionPoint`), `phase.py`, plus the
  DL/interface layer:
  `observation.py` (`build_observation` → a per-seat `Observation`: full public
  tableaus, own hand, legal actions, and a **history filtered for partial
  observability** — opponents' buried/discarded card identities are hidden) and
  `record.py` (`GameRecord`: a chess-PGN-style replayable notation — setup + ordered
  decks + every decision, with power-granted decisions in `extras`; `replay()`
  re-simulates deterministically without any RNG).
  Purchases are paid the cheapest way unless an `Action` carries a chosen `payment` (left, right)
  split (`Action.paying`, options from `Game.payment_options`; not part of action equality, so the
  action space is unchanged); the record stores such moves as `[code, card, left, right]`.
  The base-game wonder powers are real effects with their own hooks: `PlayLastCard`
  (Babylon B), `FreeBuildPerAge` (Olympia A, played as `ActionKind.FREE`, offered only
  when it saves coins), `CopyGuild` (Olympia B, best neighbour guild at scoring, incl.
  science) and `BuildFromDiscard` (Halicarnassus; resolved after the turn, so it sees
  that turn's discards and the end-of-Age leftovers).
- **`wonders/agents/`** — the agent layer over the engine. `base.Agent` is the shared
  `decide(obs) -> Action` interface. `baselines.py` has `RandomAgent`, `GreedyAgent`,
  `DiscardAgent`, and colour-focused `ArchetypeAgent`s (`MilitaristAgent`,
  `ScientistAgent`, `CivicAgent`, `EconomistAgent`, `BuilderAgent`) — cheap, diverse
  opponents for self-play. `heuristic.HeuristicAgent` is the stronger baseline: a
  tunable value function over the Observation (VP, *marginal* science-set gain,
  Age-scaled military, economy, chaining tempo, wonder progression) that builds-first
  and clearly beats greedy and builder (weights tuned with `bench.py`). `strategies.py` has characterful policies:
  `OpportunistAgent` (switches colour focus every few turns), `BlockerAgent`
  (denies the downstream neighbour their most-wanted card — Ages I/III pass left,
  Age II right), `TableAgent` (best affordable by a fixed value lookup table), and
  `BalancedAgent` (fills the scarcest colour). `board.py` has the **wonder-board-aware**
  agents: `BoardAwareAgent` (HeuristicAgent + values board powers such as Babylon's 7th
  card / Olympia's free build by Ages remaining, plans production for the resources its
  unbuilt stages need — the biggest gain — and spends the card the downstream neighbour
  wants most) and `BoardFocusAgent` (also favours the colours its board's stages point
  to). They read `Observation.wonder_board` (whole board: starting resource, stages with
  cost/effects/built, passive powers) and `Observation.production`; effect summaries
  include the power hooks (`plays_last_card`, `discard_builds`, ...). Board agent vs heuristic +
  greedy: ~62% wins (heuristic ~26%); the bundled neural agent still beats it. All read only the `Observation`, so
  they respect partial observability. `encoders.py` maps an `Observation` to a
  representation (`DictEncoder` structured, `FlatEncoder` small fixed-length vector with
  optional history, `CardEncoder` multi-hot hand + tableaus over the card vocabulary);
  `worlds.sample_world(obs, rng)` rebuilds a playable `Game` from one seat's Observation, sampling
  everything hidden (opponents' hands, hidden discards / burials, guild / black draws, leaders), so
  search cannot leak information; `Game.age_steps(first_turn=...)` lets it resume mid-Age.
  `match.play_match` runs agents and returns `(Game, GameRecord)`. `bench.run_bench` plays a
  lineup of picklable `AgentSpec`s (registry `AGENTS`) over many seeded games with rotating
  seats, reporting win rates with 95% Wilson intervals, mean score/rank and category means
  (plus per-game ranks). The `HeuristicAgent` family drafts / recruits leaders by
  `leader_values.LEADER_VALUES` — VP of margin per leader, regressed from 40k games with
  random leader play (`python -m wonders.agents.leader_values` regenerates it); the other
  scripted agents keep the base `Agent` cost proxy (≈ random). `ablation.LeaderAblation` swaps an
  agent's leader decisions for random / cost-proxy ones; PPO pools accept `name[mode]` for it
  (e.g. `board[cost]`, the pre-Stage-2 behaviour).
- **`wonders/env/`** — training environments. `ActionSpace` is a fixed discrete space
  (card name × `ActionKind`, index = `card * 4 + kind`) with `mask(legal)` /
  `to_action(index, legal)`. `WondersEnv` is a PettingZoo-`ParallelEnv`-style
  multi-agent env (no dependency): `reset(seed)` / `step({seat: index})` return
  per-acting-seat `{'observation', 'action_mask'}` (raw `Observation` in
  `infos[seat]['obs']`), rewards `margin`/`win`/`score`/`dense`, and `record()` gives
  the `GameRecord`. `SingleAgentEnv` is the Gym-style wrapper (one learner seat,
  `Agent` opponents). With Leaders the env also yields `draft` / `recruit` decisions
  (`CardEncoder(..., LEADERS_DECISIONS)` is chosen automatically). `vector.py` batches
  `SingleAgentEnv`s Gymnasium-style with auto-reset: `SyncVectorEnv` (in-process) and
  `SubprocVectorEnv` (worker processes; `envs_per_worker` amortizes lockstep rounds —
  ~1 worker per physical core). Build envs from a picklable `SingleAgentEnvFactory`
  (deep-copies opponent factories per env).

- **`wonders/play/`** — local human play (stdlib only). `session.HumanSession` wraps a
  `WondersEnv` with exactly one human seat: agent seats are played out until the human must
  decide, and `state()` is a JSON view built only from the human's `Observation` (final scores
  of all seats once done). `GameConfig` is what the menu picks (expansions, seats, wonder/side,
  seed); neural opponents are checked against their checkpoint vocabulary. `server.py` serves
  `static/index.html` (vanilla JS single page) and a JSON API on 127.0.0.1 (`wonders gui`).
  Every step of every seat is kept in `HumanSession.log`; `saved()` is the game file (config + seed,
  log, record, final `result` facts) and `restore()` replays the log on the same seed (not the agents),
  so saved games survive agent changes. The server saves after each move to `wonders.storage.Store`
  (`$WONDERS_HOME/games`, default `~/.wonders`; tests point it at a temp dir). `rating.py` holds the
  agents' fixed Elo (`AGENT_ELO`, fitted by `python -m wonders.play.rating`) and derives the human's
  rating from the finished games (pairwise vs every opponent); `achievements.py` has feats (predicates
  over a saved game's `result` facts) and career goals (progress over all finished games + ratings) —
  both derived, never stored, so new ones count old games; new facts go into `HumanSession.result()`.
  Agents keep their ids; `agents.bench.AGENT_NAMES` gives the display names (GUI, seat names, CLI aliases).

- **`wonders/ml/`** — the optional learned agent (PyTorch, `[ml]` extra; nothing else
  imports it, so the core stays dependency-free and `tests/test_ml.py` skips without
  torch; a separate `ml` CI job installs CPU torch). `model.PolicyValueNet` is a masked
  policy + value MLP over `CardEncoder` features with stored input normalization;
  checkpoints carry the card vocabulary. `agent.NeuralAgent` plays it (picklable via a
  checkpoint path) — every decision incl. the Leaders draft/recruit goes through the network
  when the checkpoint knows those kinds (`leader_policy=False` restores the old cost-proxy
  routing; before Oct 2026 evals all used it and understate Leaders by 0.5–0.9 win × n).
  `search.SearchAgent` is network-guided lookahead: the policy's top candidates are played out in
  shared sampled worlds for `depth` turns, then the value head scores them (3 turns / 16 worlds
  ranked above the plain network in ~60% of games, old rules; `ROADMAP.md`).
  `analysis.analyse(record, seat)` (`wonders analyse`, the GUI's "Analyse my moves") replays a record
  with `GameRecord.replay_observations` (each decision as the seat saw it) and rates every regular
  turn: `SearchAgent.evaluate_detailed` scores the played move next to the policy's top candidates
  in shared worlds (paired loss + standard error, labels, score-category reasons); `analysis.patterns`
  finds themes across the game, `analysis.aspects` gives per-aspect statistics for the charts, and what-if continuations replay key moments with the better move
  from the real position via `GameRecord.branch` (paired, later Ages reshuffled).
  `expert.collect_search` / `expert.distill` (`wonders train expert`) record the search's soft targets
  (softmax of candidate values) with one searching seat per game and fine-tune the network on them
  (expert iteration; ROADMAP item 2).
  `evaluate.evaluate` scores vs scripted lineups; `evaluate.head_to_head` / `compare_agents`
  (`wonders train h2h`) seat models at the same tables and report pairwise "ranked above"
  rates — the primary metric now, since seed variance dwarfs what the scripted eval sees.
  `diagnostics.leaders_report` (`wonders train leaders`) gives teacher agreement per decision
  kind and leader-decision ablations. `imitation.py` records teacher (`HeuristicAgent`) decisions in worker
  processes and clones them; its value target is the remaining score margin, i.e. the
  return of the env's `dense` reward × `VALUE_SCALE`, which `ppo.py` (PPO on
  `SubprocVectorEnv`, learner seat rotating) also uses. `pretrained/ppo_universal_v2.pt` is
  the bundled model (`pretrained_path()`, CLI `-a neural`, the GUI's neural opponent): the
  Stage 3 256×2 `table-v2` model, universal over expansions and 3–7 players (`sweeps/STAGE3.md`), continued
  with learning-rate decay (`s5-anneal`, `sweeps/OBJECTIVES.md`). **Never fork after torch has
  run**: `wonders/_mp.get_context()` switches pools to forkserver/spawn once torch is
  imported and `init_worker` pins workers to one thread — use these for any new pool.
  **Training speed**: PPO is bound by game simulation in the env workers (building
  Observations for every acting seat), not by the network. The learner runs on CUDA when
  available (`PPOConfig.device`), which mostly frees CPU cores for the workers; envs return
  compact NumPy arrays; `PPOConfig.pipeline=2` steps two env groups in turn so simulation
  and inference overlap. Use ~16 workers on a 6-core/12-thread CPU (e.g. 64 envs, 4 per
  worker). Resume counts learner `steps`, so a run can resume with a different batch size.
  Engine caches that keep this fast: payments memoized by resource cost, per-player
  aggregate caches, `Game.cached_totals()` per tick (engine-driven paths only).
  **Encoders**: `Vocab.encoder_kind` picks `table-v2` (the default, `DEFAULT_ENCODER`, for new
  models, PPO, sweeps and the CLI) or `card-v1` (`CardEncoder`, 851 inputs: own city +
  neighbours; still the raw `WondersEnv`'s own default when no encoder is passed) — `table-v2` (`agents/table_encoder.TableEncoder`, 3,020 inputs: every city
  seat-relative and padded to 7, rich hand / private / context / history blocks; design in
  `docs/encoder-v2.md`). Checkpoints store `encoder` / `decisions` / `wonders`
  (`Vocab.meta()`); checkpoints without them load as v1, so `save_checkpoint` refuses a
  non-v1-shaped model saved without `**vocab.meta()`. v2 relies on extra observation
  fields (`CityView` tokens / debt / production / score_breakdown …, `Observation`
  leaders_hand / seen_cards / buy_prices …) — keep them free of hidden information
  (`tests/test_table_encoder.py` checks opponents' hands never change the encoding).

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
- **Performance caches have invariants** (games must stay bit-identical —
  `tests/test_determinism.py` holds a golden digest; only update it for deliberate
  rule/agent changes): `Player._cache` (production, trade prices, free-build allowance)
  is cleared only by `add_card`/`add_stage`, so never mutate `cards`/`stages_built`
  directly; `Game._payments` is keyed by the seat's and neighbours' `Player.version`
  plus coins; `ObservationBuilder` reuses the public table per `Game.tick`, which the
  engine bumps on every state change it makes. Hot enums derive from `IdentityEnum`.

**Extensions** (`wonders/extensions.py`, guide `docs/extensions.md`): `REGISTRY` takes expansions,
effect types, agents, models and locales and writes through to `EXPANSIONS`, `EFFECT_BUILDERS` and
`AGENTS`. Sources are discovered once by `ensure_loaded()` (called by `load_content`, `Ruleset.of` on
unknown ids, the CLI lineup and the GUI menu): data folders under `$WONDERS_EXTENSIONS` /
`~/.wonders/extensions` (manifest + JSON / `.pt`, never code) and entry points in the
`wonders.extensions` group (code). `RecordBuilder` pins each extension's version and content hash
in `GameRecord.extensions`; replays warn on a mismatch. Tests disable discovery
(`tests/conftest.py`, `WONDERS_NO_EXTENSIONS=1`). `$WONDERS_EXTENSIONS` may list several folders
(`os.pathsep`); a plugin's `MANIFEST` is read from its `register` function's module (`plugin_manifest`).

**Contributions** (`wonders/contrib.py`, `CONTRIBUTING.md`): data extensions live in top-level
`extensions/<id>/` (folder = manifest id), pip-installable plugins in `plugins/<name>/` (`plugins/example`
is a working one; its tests are outside `testpaths` and run in CI against the installed plugin; `make
format/lint` cover `plugins/`, mypy does not), translations of the game in `wonders/i18n/<lang>.json`
(`de.json` is the reference key set). `wonders ext check` = `contrib.check` (extension / plugin / locale
reports with ok / warning / error findings; smoke games are replayed from their records);
`wonders ext changed` maps a diff to targets and restricts names to `contrib.NAME` (they reach workflow
shells); `ext scorecard` = `contrib.scorecard` (`head_to_head` vs `bundled`, shards merged by
`evaluate.merge_head_to_head`), `ext summary` renders the PR report, `ext leaderboard` updates
`leaderboard.json` + `LEADERBOARD.md`. Workflows: `contributions.yml` (pull_request, read-only; the base
branch is installed as referee, only contributed files come from the PR), the reusable `scorecard.yml`
(model × combination matrix), `contributions-comment.yml` (workflow_run; comments only if the PR head
matches) and `leaderboard.yml` (on merge; opens a PR).

**Translations** (`wonders/i18n/`): flat catalogs keyed by the English text (`de.json`; `"text|context"`
keys for words with two meanings, `_decimal` / `_ordinal` / `_language` meta keys), extended by extension
locales. `Translator(lang)` translates and formats; text stored for later rendering is a `msg(...)`
message (the analysis keeps `reason_msg` / `title_msg` / `detail_msg`, `analysis.localize`). The renderer,
`HumanSession.state/card_info/boards`, `menu_options` take a translator (English default — English
output must stay byte-identical); the server takes `?lang=` on every route and `/api/locale` gives the
page its catalog (`t()` / `tn()` / `nm()` in `index.html`, `N('…')` marks text translated through a
variable). Names stay English ids everywhere else. `tests/test_i18n.py` fails on any GUI, card or
analysis text without a German translation and on placeholder mismatches.

`wonders/wonders.py` is now a thin public-API re-export (`from wonders.wonders import
Game, Player, ...`).

**Expansions are optional and combinable** (`wonders/ruleset.py`). A `Ruleset.of('leaders',
'cities')` composes any set of expansion ids (`base` always implied); combining is a
union. Content merges via `load_content('base', 'leaders', 'cities')` — each expansion is
one `data/<id>.json` file. Rule tweaks are designed to be inert when the expansion is
absent, so no special-casing: Leaders only raises the starting treasury (`Ruleset.starting_coins`,
threaded through the `GameRecord`); Cities adds `Player.debt`/`Player.diplomacy` (both default
0) consumed by scoring (debt = −1 VP) and military resolution (a diplomacy token removes a
player from the conflict ring). Cities' new effects use the generic `Effect.on_build(ctx)` mutation hook: `Debt`,
`Diplomacy`, `Plunder` (others lose coins, overflow → debt), `Grant` (others gain
coins), `GainToken` (immediate military token). `Tally` also gained `coins`
(per 3 coins) and `victory_tokens` (with an optional `token_value` age filter)
metrics. Leader (white) cards are drafted separately so `build_age_deck` excludes
them (`Content.leaders()`); Cities (black) cards carry **no player-count numerals** —
`build_age_deck` draws `n_players` random black cards per Age (one extra card per
player, so Cities deals 8 not 7). `armada` is registered but flagged
`implemented=False` (`Ruleset.unimplemented`). Babel is not planned.

**Wonder Pack** (`wonderpack.json`, id `wonderpack`): boards may set `any_order` (Great Wall),
`extra_coins` (Manneken Pis, via `Player.starting`), `requires` (Abu Simbel: `load_content` drops it
without Leaders) and stages may `copy` a neighbour's stage. `Game.__init__` resolves copies into
`Player.slots` (per board stage, the stages it may be); `Player.stage_options()` lists what a WONDER
action may build and `Action.staged(k)` picks one (not part of equality, like `paying`; `None` = first
affordable). Records store it as a fifth move entry. `Player.built_slots` / `built_variants` /
`stage_cards` / `entombed` track the board; `CityView.stage_slots` / `entombed` expose it (worlds rebuild
from them). Effects got hooks `buried(card)` (Stonehenge B binds the buried card's colour, scored only
when `Context.final`), `leader_entombs()` (power decision `ENTOMB_LEADER`) and
`discard_build_priority()` (the FAQ order; `_activate(source=...)`). `Context.debt_tokens` is true only
with Cities: plunder without it creates no debt.

`Color` is an open, interned class (not an `Enum`): the nine official colours are constants
(`OFFICIAL_COLORS`, fixed order), and an expansion file's `"colors"` entries define more
(`Color.from_spec`, via `content.define_colors` before its cards are built). A colour carries
`label`, `fill`, `score` (the `ScoreBreakdown` column its cards' VP go to) and `tradeable` (neighbours
may buy its production, as for brown/grey). Encoders keep one feature per *official* colour only.

**Meta rules** (`ruleset.RULES` / `Rules`): starting and discard coins, coins per treasury point,
science set bonus, military token values, extra guilds. An expansion sets (`n`) or changes
(`{"add": n}`) them via `Expansion.rules` (extension files: `"rules"`); `rules_of(expansions)` applies
them in `rule_order` (base, `BUNDLED`, extensions by id — never the chosen order; content order is
untouched), clamps to each rule's range, and `Game.rules` holds a game's own, so replays, the env and
sampled worlds agree. The module
constants (`DISCARD_COINS`, `CONFLICT_VALUE`, …) are the base values the scripted agents still use.

**Expansion builder** (`wonders/play/builder.py`, GUI page `#builder`): edits one data extension folder
per project (manifest + `cards.json` with `rules`/`colors`/`cards`/`wonders`), `check_project` pins
problems to items, `save_project` writes and `extensions.reload_folder`s it (only without errors),
`preview` renders unsaved specs (`Color.from_spec(redefine=True)` updates an edited colour in place),
`balance` plays scripted agents with and without the expansion. Effect forms come from
`cards/schema.EFFECT_SCHEMAS` (one per `EFFECT_BUILDERS` type; tests enforce it). Builder routes need a
local `Host` and a JSON content type (no cross-site writes). `load_folder` builds every card and wonder
of a folder and rolls back on failure, so a broken extension is never half-registered.

`leaders.json` (36 leaders) and `cities.json` (42 black cards) hold the **official
functional decks** (costs/effects/counts), and **every power is mechanized** — no
`Special` placeholders remain (`Special` stays available for future content). The
Leaders/Cities powers are uniform hooks too: `science_set_bonus` / `science_copies`
(scoring), `build_income(BuildEvent)`, `coins_per_victory`, `redirects_defeats`,
`resource_discount(target)` (the payment solver waives the dearest units),
`builds_free(card)`, `bank_purchases` (solver bank slots), `private_production(produced)`
(own-use slots neighbours cannot buy) and `trade_reduction`. `Context.table` gives effects
every player, so the `'others'` scope hits *all* other players (Cities plunder/grant use
it). Effects built from JSON keep the spec's `description` as `effect.label` for the card
renderer. Power decisions (Halicarnassus / Solomon discard builds, Courtesans' leader copy,
Rome B extra recruits) are resolved by `Game._power_steps`, after every turn *and* during
recruitment. Powers were checked against the official Leaders and Cities rulebooks
(Sepulcher/Cenotaph count victory tokens; Nero pays for every victory token incl. raider
tokens; Hatshepsut pays per neighbour bought from; Black Market / Secret Warehouse look only at
brown/grey cards + the board resource; masks copy distinct neighbour *green cards*). Our card
names are paraphrased (e.g. Pigeonhole = Pigeon Loft, Chamber of Builders = Builders' Union,
Architect Firm = Architect Cabinet); Smugglers Cache has no counterpart in the 1st-edition
Cities rulebook (East/West Clandestine Wharf are the official trade-rebate cards).

The Leaders draft (pick-and-pass of 4) and each Age's recruitment are engine coroutines
(`Game.draft_steps` / `Game.recruit_steps`, decision kinds `DRAFT` / `RECRUIT`) driven by
`play_and_record` (via `match.play_match`), replay and the env alike. A draft pick is
expressed as `Action.build(leader)` so the action space is unchanged; `Agent.act(obs,
options)` routes draft/recruit decisions to `Agent.draft_leader` / `Agent.recruit`.
Leaders also adds 4 guilds (Gamers, Courtesans, Diplomats, Architects — mixed into Age III's
guild draw) and Rome; Cities adds Petra and Byzantium. Costs/effects come from the official
rulebooks (Rome's and Byzantium B's stage costs were read off the rulebook board images).
`Wonder.starting_resource` may be `None` (Rome) and `Wonder.effects` holds passive board powers
(Rome A: free leaders; Rome B: `CoinDiscount` 2 for self / 1 for neighbours). Rome B stage 1
draws from `Game.leader_box` (the undealt leaders, recorded in `GameRecord.leader_box` so
replays are exact); stages 2–3 grant a `LEADER_BUILD` decision. The Courtesans' Guild grants a
`COPY_LEADER` decision; the copy lives in `Player.copied_leaders` (powers apply, but it is not a
leader card for the Diplomats' Guild). `ScoreBreakdown` has `leaders` (white cards + copies)
and `cities` (black cards) columns; the official scorecard adds both on one row.

## Conventions

- `black` and `isort` are authoritative for formatting (line length 120, `skip-string-normalization`). Google-style docstrings; `flake8` ignores `E203, E266, W503`.
- Type annotations are expected — `mypy` runs in the lint gate (over `wonders` and `tests`).
- Python **3.11+** (`requires-python` in `pyproject.toml`).
- Pre-commit hooks (`.pre-commit-config.yaml`) enforce the above plus CRLF/whitespace fixes; run `make pre-commit` to check all files.
