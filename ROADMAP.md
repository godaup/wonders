# Roadmap

Planned work, roughly in priority order. Each item says why it matters, what is already in place,
and what is left. Finished work moves to `CHANGELOG.md`; training reports live in `sweeps/`.

## 1. Retrain after the October 2026 rules fixes

**Status: done** (October 2026, `sweeps/RETRAIN.md`): leader values re-estimated; `r1-a`, fine-tuned
from `s5-anneal`, is bundled (52.2% head-to-head over it; rules version 2). The search re-check
moves to item 2.

**Why.** Three engine bugs were fixed in October 2026 (PRs #28 and #29):

* neighbours may only buy the board's resource and brown / grey card production — yellow
  cards (Forum, Caravansery, …), wonder stages and leaders produce for their owner only;
* a card used to build a wonder stage stays under the board instead of going to the discard
  pile;
* a discarded leader returns to the box instead of going to the discard pile, so Halicarnassus /
  Solomon can only build discarded Age cards.

Every trained model up to and including the bundled `s5-anneal`, the leader value table and all
results in `sweeps/` (including the search experiments) predate these fixes. The models still play
legal games, but they misjudge trading (they expect to buy goods that are no longer for sale) and
the discard pile.

**Plan.**

1. Regenerate `leader_values.LEADER_VALUES` under the new rules
   (`python -m wonders.agents.leader_values`) and re-check the scripted agents' weights
   (`HeuristicAgent`, `board`, `adaptive` — the adaptive agent already uses the neighbours'
   tradeable production).
2. Fine-tune the bundled model under the new rules: a PPO continuation of `s5-anneal` with the
   same opponent mix and a linear learning-rate decay (the `s5-anneal` recipe, ~10M steps).
3. Evaluate with random seating against the current bundled model and the strongest scripted
   agents (all expansions, 3–7 players), then bundle the result if it is at least as strong.
4. Re-run a short search check (below) so later comparisons use the new rules.

## 2. Network-guided search and expert iteration

**Status:** search and the expert-iteration pipeline (`wonders train expert`) are implemented.
Under the current rules search still beats the bundled `r1-a` (56.9%), but distilling flat search
into the network did not help: eight distillation variants and a PPO polish all land at or below
`r1-a` (best 50.1%; `sweeps/EXPERT.md`). Flat-search expert iteration is shelved until a better
teacher exists (step 3 below).

* `wonders/agents/worlds.py` rebuilds a playable game from one seat's observation and samples the
  hidden cards (opponents' hands, hidden discards / burials, the guild / black-card draws,
  opponents' leaders), so search cannot use hidden information.
* `wonders/ml/search.py` (`SearchAgent`) plays the network's top candidates out in shared sampled
  worlds for a few turns, then uses the value head.
* Old-rules results against plain `s5-anneal` at the same tables (800 games each): 3 turns ahead
  with 16 worlds ranks above it in 60.2% of games; 2 turns / 32 worlds 58.0%; full rollouts are
  worse than no search (too noisy). Gains seem to level off around 56–60%.

**Next.**

1. ~~Depth / width sweep, re-check under the new rules, merge~~ — done.
2. ~~Expert iteration with flat search~~ — done, no gain (`sweeps/EXPERT.md`).
3. **A better teacher:** information-set MCTS or decoupled UCT for the simultaneous moves (alpha-beta
   does not fit: 3–7 players, simultaneous moves, hidden cards); its visit counts give sharp
   targets for the existing distillation pipeline. Opponent modelling for the determinisation
   (Further ideas) reduces the noise of any sampled-world search.

## 3. Post-game analysis

**Status: done** — `wonders analyse` and the GUI's "Analyse my moves": move ratings with paired noise
estimates, key moments with reasons, the expected-margin graph, patterns across the game, and what-if
continuations of the key moments from the real position, and per-aspect ratings and charts
(category ranks, points lost per move, accuracy by Age and move type, military, economy) (`wonders/ml/analysis.py`,
`GameRecord.replay_observations` / `GameRecord.branch`). Possible follow-ups: rating the Leaders draft
and recruitment, and analysing several games for trends.

**Goal.** After a game in `wonders gui`, an "Analyse game" button explains how the human played,
like a chess engine's game review.

**What it shows.**

* **Move ratings** — for every human decision, every legal move is evaluated and compared with the
  one played; the gap in expected VP margin classifies it (about equal / inaccuracy / mistake /
  blunder), e.g. *"Age II, turn 3: discarding Caravansery cost ~4 VP — Forum would have covered the
  glass you later bought for 6 coins."*
* **Evaluation graph** — the human's expected final margin (or win chance) after every turn, with
  the turning points marked.
* **Key moments** — the 3–5 costliest decisions with the hand, the move played, the recommended
  move and why (VP, coins saved, military, science sets, chains, the card passed to the neighbour).
* **Patterns** — recurring themes aggregated over the game: coins spent on trade, military lost by
  one shield, unfinished science sets, strong cards passed to a neighbour, unused board powers.
* **What-if replays** — replace one move with the recommended one and play the rest out many times
  with the agents ("1st in 58% of continuations instead of 22%").

**Building blocks already present.** `GameRecord` replays every position exactly, so each decision
can be rebuilt with the human's own observation; the search agent's world sampling and rollout
evaluation is exactly the evaluator needed; the value head gives the evaluation graph cheaply.

**Design rules.**

* Judge decisions with the information the player had (sampled worlds), not hindsight; an optional
  hindsight view may reveal the true hands.
* Report confidence: differences below ~1 VP are "about equal"; flag only clear mistakes.
* Run as a background job with progress (about 1–4 s per decision on a CPU, ~1 minute per game;
  parallel processes for speed); the report is JSON, so it can be cached and downloaded.

**Prerequisites.** Item 1 (a model trained under the current rules) and item 2 (the search
evaluator merged).

**Plan (~2–3 days).** Analysis job + JSON report (move ratings, key moments) → GUI review screen
(graph, key moments with hand / board, patterns) → what-if replays.

## 4. Model challenge via pull requests

**Status: built, broadened to every kind of contribution, not yet exercised by a real pull request**
(`wonders/contrib.py`, `.github/workflows/contributions.yml`, `scorecard.yml`,
`contributions-comment.yml`, `leaderboard.yml`). Models, expansions and extension locales go to
`extensions/<id>/` (data), plugins to `plugins/<name>/` (pip packages, with `plugins/example`),
translations of the game to `wonders/i18n/`; each kind is checked by `wonders ext check`, and models
get the scorecard below, a PR comment and a leaderboard PR after the merge. Still open: the
play-style statistics in the scorecard (coins spent on trade, wonder stages, science sets — the
head-to-head does not collect them yet); enabling "Allow GitHub Actions to create pull requests" for
the leaderboard workflow; a first real contribution to calibrate the run time (~5,000 games in four
jobs).

**Goal.** Anyone can submit a model by pull request and get an automatic head-to-head against the
bundled network, with a scorecard posted on the PR — the README's "beat the machine" challenge as a
contribution path, ending in a public leaderboard.

**Submission.** A PR adds an extension folder `extensions/<id>/` with `manifest.json` (author,
licence, description, rules version, the model's name and how it was trained) and the checkpoint (a
256×2 checkpoint is ~4 MB, no Git LFS needed).

**Evaluation workflow** (GitHub Actions, triggered by changes under `extensions/`):

1. **Validate** before spending compute: a loadable checkpoint (loaded with `weights_only=True`, as
   everywhere), the expected encoder and vocabulary, a size limit, and the rules version it was
   trained under — checkpoints should start recording it, so models from before a rules change are
   flagged.
2. **Play** `wonders train h2h <name>=models/contrib/<name>.pt bundled` with a fixed seed and random
   seating, split into a job matrix by expansion combination (4 cores each on a standard runner;
   ~4,000 games in roughly 10–20 minutes, about ±1.5–2 points of 95% interval).
3. **Report** as a PR comment and job summary:
   * overall: ranked above the bundled network (with interval), first-place share, mean score;
   * per expansion combination and player count, showing where the model is stronger or weaker;
   * strengths and weaknesses by score category (military, science, commercial, guilds, wonders,
     treasury) relative to the bundled network, plus play-style statistics (coins spent on trade,
     wonder stages built, science sets completed) — the head-to-head report needs to collect these;
   * a verdict: *new state of the art* only when the interval's lower bound is above 50%.
4. **Leaderboard** — on merge, a workflow updates `LEADERBOARD.md` (or a GitHub Pages page); a clear
   winner can become the next bundled network.

**Security.** The evaluation runs main's code and takes only the model file from the PR, so a PR
cannot change the referee. Use the two-workflow pattern: the `pull_request` job (read-only token,
no secrets) evaluates and uploads the results as an artifact; a separate `workflow_run` job with
permission to comment posts them without executing anything from the PR.

**Reproducibility.** Fixed seeds: the scorecard lists the exact command, so anyone can reproduce
the numbers locally.

**Prerequisites.** GitHub Actions running for the repository (free for public repositories on
standard runners); a rules-version field in checkpoints. **Effort:** ~1–2 days, mostly the
scorecard, the category breakdown and the validator.

## 5. Remaining expansions

Expansions are optional and combinable (`wonders/ruleset.py`); content is data in
`wonders/cards/data/<id>.json`, new mechanics are new `Effect` hooks, and turn-structure changes are
engine coroutines like the Leaders draft. Leaders and Cities are complete. For each expansion below,
the work is: transcribe the official card / board data, mechanize its new rules and effects, expose
the new public state in the `Observation` (without hidden information), extend the encoders and
action space, add agent heuristics, and train a network whose vocabulary covers it.

The summaries below are an outline from general knowledge of these expansions; check every rule
against the official rulebook before implementing it (as was done for Leaders and Cities).

### Armada (registered, `implemented=False`)

* Naval boards attached to each wonder board with four ship tracks (red, yellow, blue, green):
  building a card of a colour can advance that colour's ship by paying the track cost.
* Naval conflicts at the end of each Age (a second military comparison over the naval track,
  against every player, not just neighbours) with their own victory tokens.
* Island cards explored via the blue track; new Age cards (shipyards and others) and new guilds.
* Engine: a per-build optional "advance a ship" payment, the naval conflict resolution, island
  decks; observation / encoder: ship positions per seat, naval strength, islands owned.

### Wonder Pack (done, October 2026)

The four boards of the official Wonder Pack (`wonderpack.json`), checked against its rules sheet and
the board images: the Great Wall (stages in any order, chosen with `Action.staged`), Abu Simbel
(entomb a recruited leader: its power is lost, twice its cost in VP; Leaders only), Stonehenge (stone
symbols on brown cards; side B scores the buried card's colour at the end) and the Manneken Pis
(copies its neighbours' stages, cost included; +4 coins). The FAQ's discard-build order (Halicarnassus,
Great Wall, Manneken Pis, Solomon, Courtesans) applies to every game. Open: agents that plan the Great
Wall's stage order (they build the first affordable one), and a network trained with the new boards
(the bundled one refuses them).

### Custom expansions

Content is already data, so new cards and boards need no engine changes as long as they use the
existing effect types. A custom-expansion feature would make that a first-class workflow.

**Done:** loading expansions through the extension system (item 6) with `wonders ext validate |
check`; **new card colours** declared in the expansion file (`"colors"`: label, fill, score row,
tradeable production); **meta rules** an expansion may override (`"rules"`: starting / discard coins,
coins per point, military tokens, science set bonus, guild count; `wonders/ruleset.py`); **effect
schemas** for every effect type (`wonders/cards/schema.py`, plugins pass one to `register_effect`);
and the **expansion builder** in `wonders gui` (`wonders/play/builder.py`): colours, cards and both
sides of wonder boards in forms with live previews, meta rules, per-item checks, save-and-play, a
balance check against the base game with scripted agents, and zip export. New *mechanics* stay plugin
code (Python); the builder composes existing effect types.

Next for the builder: running the full `wonders ext check` report from the page, importing a zip,
translations of an expansion's card names (a locale file per extension), and generating the effect
table in `docs/extensions.md` from the schemas.

* **A documented effect catalogue** — every effect type with its parameters and an example card,
  generated from `EFFECT_BUILDERS`, so designers can compose new cards without reading code; a
  validator that reports unknown effects, broken chains and deck sizes that do not match the
  player counts.
* **Card designer preview** — the card renderer and the GUI's tooltips work for custom cards
  unchanged; a `wonders gallery -e my_expansion.json` page shows the whole set at a glance.
* **Agents that cope with unseen cards** — scripted agents read effects through the uniform
  hooks, so they play custom cards out of the box. The network's encoder indexes cards by name;
  an effect-based card encoding (cost, production, points, science, chains as features) would let
  a model generalise to cards it was never trained on, and is worth having anyway.
* **An original expansion of this project** — once the workflow exists, a small official set of
  new cards and a board designed and balanced with the agents: simulate thousands of games to
  check that no card or strategy dominates before anyone plays it.

### Out of scope for now

* **7 Wonders, 2nd edition** (2020) — different card set, Edifice expansion and some rule changes;
  it would be a separate ruleset rather than an expansion of this one.
* **7 Wonders Duel** — a different two-player game.
* **Babel** — dropped (October 2026); custom expansions get the effort instead.

## 6. Extensions: custom expansions, agents and models

**Status: steps 1, 2 and 4 done** (`wonders/extensions.py`, `docs/extensions.md`): one registry for
expansions, effects, agents, models and locales; data folders and pip plugins (entry point
`wonders.extensions`); manifests with content hashes pinned in game records; `wonders ext list |
validate | new | check`; curated pull requests to `extensions/` and `plugins/` with automatic checks
(item 4). Still open: the Hugging Face source (step 3).

**Goal.** Let anyone add expansions, agents and trained models without changing this repository,
with one mechanism and clear trust rules.

**Two kinds of extension.**

* **Data** — card / wonder JSON (expansions built from existing effect types), network checkpoints,
  game records and datasets. Safe to load automatically after validation (checkpoints already load
  with `weights_only=True`).
* **Code** — new effect types (new mechanics), scripted agents, new decision kinds. Arbitrary Python:
  it needs the same trust as any `pip install`.

**Design.**

1. **One registry in the core** — `register_effect`, `register_agent`, `register_expansion`,
   `register_model`. The built-in content, agents and the bundled network register through the same
   API, so there is a single path for everything.
2. **A manifest per extension** — id, name, version, author, licence, the rules version and engine
   API it was built for, what it contains, and a content hash.
3. **Sources feeding the registry:**
   * **Local folders** (e.g. `~/.wonders/expansions/`, `~/.wonders/models/`) — data only; offline and
     good for private experiments.
   * **Python plugins** — `pip install wonders-myexpansion`, discovered through standard entry points
     (`wonders.effects`, `wonders.agents`, `wonders.expansions`, `wonders.models`); the place for code.
   * **Hugging Face Hub** (optional extra) — models and datasets with model cards, e.g.
     `-a neural:hf://user/model`; data only, never code.
   * **Curated** — the repository itself via pull requests: GitHub Actions validate a submission and
     simulate it (balance for expansions, the head-to-head of item 4 for models) before it is merged
     and bundled.
4. **Trust rules** — data loads automatically after validation; code only through an explicit
   `pip install`; the game never downloads code by itself.
5. **Compatibility** — game records store each extension's id, version and hash, so a game replays
   only with identical content; extensions declare the rules version they target; networks refuse
   unknown cards until the effect-based card encoding (custom expansions above) exists.
6. **Tooling** — `wonders ext list | validate | new` (scaffold an expansion or agent package); the
   GUI menu lists installed expansions, agents and models; `wonders gallery` previews an expansion.

**Not now:** a sandboxed scripting layer (a custom language or WebAssembly) for mechanics without
Python. The JSON effect specs already act as a small language for most cards, and the effect
catalogue extends it; revisit only if non-programmers start designing mechanics.

**Order and effort.**

1. Registry, manifests and local folders (~2–3 days) — the foundation.
2. Python plugins via entry points (~1 day).
3. Hugging Face models and datasets (~1 day).
4. Curated PRs with GitHub Actions (builds on item 4, ~1–2 days more).

## 7. Translations

**Status: done for German** (`wonders/i18n/`, `docs/extensions.md#translations`): the GUI, the card and
board renderer, card tooltips and the post-game analysis, with a language picker (browser language
by default, `?lang=` in the URL) and `wonders gallery --lang`. Catalogs are keyed by the English
text; a test fails on any untranslated GUI or card text. Still open: CLI messages, the documentation,
fonts / right-to-left for non-Latin scripts.

**Goal.** Play and read everything in your own language.

* **What gets translated:** the GUI (menu, status bar, tooltips, score sheet), the card renderer's
  texts (categories, effect descriptions, cost labels), CLI messages, and later the documentation.
* **How:** one locale file per language (`wonders/locales/<lang>.json`) with keyed strings and
  placeholders; a small `t(key, **values)` helper in Python and its counterpart in the page; card
  and wonder names and effect descriptions keyed by their canonical id. A language picker in the GUI
  menu, `--lang` on the CLI, the system language as the default.
* **Stays canonical:** game records, agent and checkpoint vocabularies and the API use the English
  ids, so a game recorded in one language replays in any other.
* **Names:** this project's card names are its own paraphrases and are translated as such; the
  publisher's official localized names and texts are not shipped (see the disclaimer).
* **Rendering:** fonts for non-Latin scripts in the card SVGs and the page; right-to-left layouts
  once a right-to-left language is added; text wrapping already adapts to longer translations.
* **Contributions:** language packs are data, so they fit the extension system (item 6); a
  completeness check in CI lists untranslated keys.

**Effort:** ~2–3 days for the mechanism plus English and one more language; each further language is
a data contribution.

## 8. Table companion: from a real table to a move proposal

**Goal.** Bring the engine to games played elsewhere — a physical table or a recorded game: turn
the position into an `Observation`, ask the network (or search) for a move, and explain it.

**One core, several front-ends.**

1. **Position editor (the core).** Enter or correct a table state in the GUI: Age and turn, your
   hand, every city's cards, coins, wonder stages and military tokens, Leaders and Cities state. It
   builds an `Observation`; `worlds.sample_world` already turns that into playable games with the
   hidden cards sampled, so the search agent can recommend a move, with the explanation from
   post-game analysis (item 3). Useful on its own for studying positions.
2. **Voice input.** Dictate the position or the moves ("I hold Forum, Baths and Altar; left
   neighbour built Barracks"): local speech-to-text, a parser with fuzzy matching against the card
   names (and their translations, item 7), and follow-up questions for what is missing. Optional
   spoken answers.
3. **Image input.** Photos or video frames of the table or a screen: recognise cards, boards, coins
   and tokens and fill the editor. Two routes: a multimodal model that reads the cards (fast to
   build, needs an API or a large local model), or a dedicated recogniser trained on photos the user
   provides (no publisher artwork is shipped). Always with a confirmation step in the editor —
   recognition errors must not silently change the advice.
4. **Recorded gameplay.** Step through a recorded physical or digital game turn by turn, rebuild a
   `GameRecord`, then run the post-game analysis on it.

**Fair play.** The companion is meant for learning, solo and offline play and post-game analysis. Using
it during live competitive games against people (online platforms included) is cheating and usually
against their terms; the documentation and the GUI say so clearly.

**Order and effort.** Position editor (~2–3 days, with item 3's explanations) → voice (~2–3 days) →
image recognition (~1–2 weeks, depending on the route) → recorded games.

## 9. Play in the browser, no installation

**Goal.** A static page (e.g. on GitHub Pages) where anyone can play against the bundled network —
the fastest way into the README's challenge.

* The engine is pure Python with no compiled dependencies, so it can run in the browser through
  **Pyodide** (Python on WebAssembly); the GUI's page already talks to a small JSON API, which a
  thin in-browser adapter can provide instead of the local server.
* The network (256×2) is exported to **ONNX** and runs with ONNX Runtime Web; scripted agents run as
  they are. Search is optional (slower in the browser) and could run in a web worker.
* Game records download as usual; a link can carry a record for sharing replays.

**Effort:** ~3–5 days (adapter, ONNX export and parity test, start-up size and speed tuning).

## 10. Accessibility

* **Colour-blind support:** card categories are colour-coded; add a colour-blind palette and a
  pattern or icon per category everywhere colours carry meaning (cards, chips, score sheet).
* **Keyboard play:** every action reachable without a mouse, with visible focus.
* **Screen readers:** labels for cards, actions and cities; the move log and status bar as live
  regions.
* Check contrast in both themes; respect reduced-motion settings.

**Effort:** ~2 days.

## 11. Difficulty levels and adaptive opponents

**Goal.** Opponents that match the player, from beginner to the full-strength network.

* **Tiers** from existing material: scripted agents, earlier training snapshots, the network with
  sampling instead of greedy moves (a temperature), the full network, and search on top. Each tier
  is calibrated by simulation to a target strength.
* **Adaptive mode:** adjusts the tier after each game to keep the human near a 50% win rate.
* The GUI menu offers difficulty names instead of agent names (agents stay available for experts).

**Effort:** ~2–3 days, mostly calibration runs.

## 12. Puzzles and coach mode

**Goal.** Learn the game from the engine.

* **Puzzles:** "find the best move" positions taken from simulated games where search finds a clear
  gap between the best and the second-best move; the solution is explained with the post-game
  analysis (item 3). A daily puzzle; themed sets (science, military, wonders, leaders).
* **Coach mode:** in a normal game, ask for a hint (how good is each move?) or let the coach flag
  mistakes as they happen.
* Both build on search and the analysis explanations; the position editor of item 8 lets anyone
  create puzzles too.

**Effort:** ~2–3 days after item 3.

## 13. Balance lab

**Goal.** Use millions of simulated games to answer balance questions — for strategy fans and for
designers of custom expansions (items 5 and 6).

* **Reports:** win rates per wonder and side by player count and expansions, card values by Age,
  opening tendencies, how strong each colour strategy is, how much seat position matters.
* **What-if variants:** change a rule or a card (trade price, a card's cost, a stage's effect) and
  compare against the baseline with confidence intervals.
* **Designer workflow:** a new card or wonder from an extension is simulated automatically, with a
  warning when it is clearly over- or under-powered.
* Results as an HTML report (and a page on the website).

**Caveat:** the agents' play shapes the numbers; report them per agent family and prefer the
strongest agents.

**Effort:** ~3–4 days for the framework and the first report.

## 14. Language models as players

**Goal.** A benchmark of large language models against the engine's agents — a timely showcase.

* A text interface over the `Observation` (hand, cities, costs, legal moves) and a strict answer
  format; illegal or unparsable answers count as a fallback move and are reported.
* Pluggable providers (local models and hosted APIs); prompts, settings and costs are recorded.
* A leaderboard of LLMs against scripted agents and the network, with the same head-to-head
  statistics as everything else; game records show the models' reasoning.

**Effort:** ~2–3 days, plus API budget for the runs.

## 15. Official variants still missing

* **Two players:** the base game's two-player rules with a third, shared seat played alternately by
  both players.
* **Team play** for 4–8 players (Cities), with team scoring.
* **Eight players** (supported with Cities).

These are rules, not new cards, so they fit the engine; like Leaders and Cities, each rule is
checked against the official rulebooks first. The training environment and the network need the
new seat counts and decisions (the two-player shared seat is an extra decision kind).

**Effort:** ~3–5 days for all three, plus training for the network to play them.

## Further ideas

* **Multiplayer with friends** — humans and bots at one table over the local network or the
  internet; spectator mode; shareable replay links.
* **Player profile** — partly done: the GUI keeps every game locally (`wonders.storage`) with the
  player's Elo rating over time and achievements. Still open: statistics and favourite wonders.
* **Strategy explorer** — what the network values (card heatmaps by Age, "why this move"); a
  language model could narrate the search's numbers in plain words.
* **Environment wrappers and a benchmark suite** — official PettingZoo / Gymnasium registrations and
  fixed seeds, opponents and metrics, so research results are comparable.
* **Opponent modelling for search** — infer opponents' likely hands and plans from what they played
  and passed instead of sampling hidden cards uniformly; the most direct route to stronger search
  and expert iteration.
* **Smaller, faster models** — distilled and quantized networks for phones, the browser (item 9)
  and faster search.
* **Rules oracle** — the official FAQ cases as tests, and a way to ask edge-case questions by running
  the engine.

## Smaller items

* **Python 3.14 follow-ups.** 3.11–3.14 are supported and tested (October 2026). Still open:
  * add 3.14 to the `ml` job once PyTorch and NumPy publish 3.14 wheels;
  * try the **free-threaded build** (`python3.14t`, no GIL). The engine is pure Python, so game
    simulation could run in threads instead of worker processes (less memory and no pickling in
    vector envs, benchmarks and search). It needs a thread-safety review of the shared caches
    (`Player._cache`, `Game._payments`, the observation summary cache, `lru_cache`d content) and a
    PyTorch build for free-threading; benchmark before switching anything.
* Publish to PyPI once the API settles (release flow in `CLAUDE.md`).
* A replay viewer in the GUI for recorded games (`GameRecord` JSON), reusing the review screen of
  item 3.
