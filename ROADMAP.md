# Roadmap

Planned work, roughly in priority order. Each item says why it matters, what is already in place,
and what is left. Finished work moves to `CHANGELOG.md`; training reports live in `sweeps/`.

## Done (2026)

* **Retrained under the October 2026 rules** — `r1-a` is bundled (52.2% head-to-head over
  `s5-anneal`; rules version 2), leader values re-estimated (`sweeps/RETRAIN.md`).
* **Network-guided search and expert iteration** — `SearchAgent` over sampled worlds ranks above the
  bundled network in 56.9% of games; distilling flat search did not help (`sweeps/EXPERT.md`). The
  next step is a better teacher (item 1).
* **Post-game analysis** — `wonders analyse` and "Analyse my moves": move ratings with noise
  estimates, key moments with reasons, patterns, what-if continuations, per-aspect charts.
* **Extensions** — one registry for expansions, effects, agents, models and locales; data folders
  and pip plugins; versions and content hashes pinned in game records (`docs/extensions.md`).
* **Contributions by pull request** — `wonders ext check` for extensions, plugins, translations
  and models, a scorecard against the bundled network, PR comments and the leaderboard.
* **Translations** — German for the GUI, cards, boards and the analysis, with a language picker.
* **Local storage, Elo and achievements** — saved and resumable games, ratings against agents with
  fixed Elo, 42 feats and career goals.
* **Wonder Pack** — the Great Wall, Abu Simbel, Stonehenge and the Manneken Pis, every power
  mechanized from the rules sheet.
* **Custom expansions** — new card colours, meta rules, effect schemas and the expansion builder in
  the GUI.
* **Wonder side strength** — the first balance-lab report: every side's value in VP of margin per
  expansion set and player count (`wonders.agents.wonder_values`, `docs/wonder-strength.md`).
* **Card and board strength in the builder** — a rate card while editing, paired simulations after
  saving and of the whole expansion, with recalibration (`docs/card-strength.md`).

## 1. A better search teacher

**Status:** flat search beats the bundled network (56.9% ranked above it), but its soft targets are
too noisy to distil: eight distillation variants and a PPO polish land at or below `r1-a` (best
50.1%; `sweeps/EXPERT.md`).

**Next.** Information-set MCTS or decoupled UCT for the simultaneous moves (alpha-beta does not fit:
3–7 players, simultaneous moves, hidden cards); its visit counts give sharp targets for the existing
distillation pipeline (`wonders train expert`). Opponent modelling for the determinisation (Further
ideas) reduces the noise of any sampled-world search.

## 2. A network for the Wonder Pack boards

The bundled network refuses the Wonder Pack boards (its vocabulary predates them), so games with
them need scripted opponents. Train a network whose vocabulary covers the four boards, starting from
`r1-a` with an extended vocabulary, and teach the scripted agents to plan the Great Wall's stage
order (they build the first affordable stage). The effect-based card encoding (item 4) would make
future boards work without retraining.

## 3. Balance lab

**Goal.** Use millions of simulated games to answer balance questions, for strategy fans and for
designers of custom expansions.

**Done:** the builder's balance check (scripted agents with and without an expansion) and the
strength of every wonder side per expansion set and player count (`wonders.agents.wonder_values`:
board rotation over the same decks, own and opponents' sides in one regression, partial pooling over
tables, cluster-robust errors).

**Next.**

* **Card values** by player count and expansion set (the calibration measures them pooled), and how
  strong each colour strategy is; how much seat position matters.
* **What-if variants:** change a rule or a card (trade price, a card's cost, a stage's effect) and
  compare against the baseline with confidence intervals.
* **Designer workflow:** done in the builder (rate card, simulations, recalibration); next, a warning
  in the check when an item falls clearly outside the official range, and a second agent family for the
  simulations.
* **Using the numbers:** board-adjusted Elo (the expected result given the boards dealt), balanced
  deals for benchmarks, and agent fixes where a side looks weak only because the agent misplays it
  (Petra B's 14-coin stage, see `docs/wonder-strength.md`).
* Results as an HTML report (and a page on the website).

**Caveat:** the agents' play shapes the numbers; report them per agent family and prefer the
strongest agents.

## 4. Remaining expansions and custom content

Expansions are optional and combinable (`wonders/ruleset.py`); content is data in
`wonders/cards/data/<id>.json`, new mechanics are new `Effect` hooks, and turn-structure changes are
engine coroutines like the Leaders draft. Leaders, Cities and the Wonder Pack are complete. For each
expansion below, the work is: transcribe the official card / board data, mechanize its new rules and
effects, expose the new public state in the `Observation` (without hidden information), extend the
encoders and action space, add agent heuristics, and train a network whose vocabulary covers it.

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

### Custom expansions

Content is already data, so new cards and boards need no engine changes as long as they use the
existing effect types. A custom-expansion feature would make that a first-class workflow.

**Done:** loading expansions through the extension system with `wonders ext validate |
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

## 5. Model challenge: the first real contributions

The pipeline is built (`wonders/contrib.py`, `contributions.yml`, `scorecard.yml`,
`contributions-comment.yml`, `leaderboard.yml`) but no real pull request has exercised it yet.
Still open:

* play-style statistics in the scorecard (coins spent on trade, wonder stages, science sets); the
  head-to-head does not collect them yet;
* enabling "Allow GitHub Actions to create pull requests" for the leaderboard workflow;
* a first real contribution to calibrate the run time (~5,000 games in four jobs).

## 6. Extensions and translations: what is left

* **Hugging Face Hub source** (optional extra): models and datasets with model cards, e.g.
  `-a neural:hf://user/model`; data only, never code.
* **Translations:** CLI messages, the documentation, fonts and right-to-left layouts for non-Latin
  scripts; more languages as data contributions.

## 7. Table companion: from a real table to a move proposal

**Goal.** Bring the engine to games played elsewhere — a physical table or a recorded game: turn
the position into an `Observation`, ask the network (or search) for a move, and explain it.

**One core, several front-ends.**

1. **Position editor (the core).** Enter or correct a table state in the GUI: Age and turn, your
   hand, every city's cards, coins, wonder stages and military tokens, Leaders and Cities state. It
   builds an `Observation`; `worlds.sample_world` already turns that into playable games with the
   hidden cards sampled, so the search agent can recommend a move, with the explanation from
   post-game analysis. Useful on its own for studying positions.
2. **Voice input.** Dictate the position or the moves ("I hold Forum, Baths and Altar; left
   neighbour built Barracks"): local speech-to-text, a parser with fuzzy matching against the card
   names (and their translations), and follow-up questions for what is missing. Optional
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

**Order and effort.** Position editor (~2–3 days, with the analysis' explanations) → voice (~2–3 days) →
image recognition (~1–2 weeks, depending on the route) → recorded games.

## 8. Play in the browser, no installation

**Goal.** A static page (e.g. on GitHub Pages) where anyone can play against the bundled network —
the fastest way into the README's challenge.

* The engine is pure Python with no compiled dependencies, so it can run in the browser through
  **Pyodide** (Python on WebAssembly); the GUI's page already talks to a small JSON API, which a
  thin in-browser adapter can provide instead of the local server.
* The network (256×2) is exported to **ONNX** and runs with ONNX Runtime Web; scripted agents run as
  they are. Search is optional (slower in the browser) and could run in a web worker.
* Game records download as usual; a link can carry a record for sharing replays.

**Effort:** ~3–5 days (adapter, ONNX export and parity test, start-up size and speed tuning).

## 9. Accessibility

* **Colour-blind support:** card categories are colour-coded; add a colour-blind palette and a
  pattern or icon per category everywhere colours carry meaning (cards, chips, score sheet).
* **Keyboard play:** every action reachable without a mouse, with visible focus.
* **Screen readers:** labels for cards, actions and cities; the move log and status bar as live
  regions.
* Check contrast in both themes; respect reduced-motion settings.

**Effort:** ~2 days.

## 10. Difficulty levels and adaptive opponents

**Goal.** Opponents that match the player, from beginner to the full-strength network.

* **Tiers** from existing material: scripted agents, earlier training snapshots, the network with
  sampling instead of greedy moves (a temperature), the full network, and search on top. Each tier
  is calibrated by simulation to a target strength.
* **Adaptive mode:** adjusts the tier after each game to keep the human near a 50% win rate.
* The GUI menu offers difficulty names instead of agent names (agents stay available for experts).

**Effort:** ~2–3 days, mostly calibration runs.

## 11. Puzzles and coach mode

**Goal.** Learn the game from the engine.

* **Puzzles:** "find the best move" positions taken from simulated games where search finds a clear
  gap between the best and the second-best move; the solution is explained with the post-game
  analysis. A daily puzzle; themed sets (science, military, wonders, leaders).
* **Coach mode:** in a normal game, ask for a hint (how good is each move?) or let the coach flag
  mistakes as they happen.
* Both build on search and the analysis explanations; the position editor of the table companion
  (item 7) lets anyone create puzzles too.

**Effort:** ~2–3 days.

## 12. Language models as players

**Goal.** A benchmark of large language models against the engine's agents — a timely showcase.

* A text interface over the `Observation` (hand, cities, costs, legal moves) and a strict answer
  format; illegal or unparsable answers count as a fallback move and are reported.
* Pluggable providers (local models and hosted APIs); prompts, settings and costs are recorded.
* A leaderboard of LLMs against scripted agents and the network, with the same head-to-head
  statistics as everything else; game records show the models' reasoning.

**Effort:** ~2–3 days, plus API budget for the runs.

## 13. Official variants still missing

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
  player's Elo rating over time and achievements. Still open: statistics and favourite wonders (the
  wonder-side strengths of the balance lab give a baseline to compare them with).
* **Strategy explorer** — what the network values (card heatmaps by Age, "why this move"); a
  language model could narrate the search's numbers in plain words.
* **Environment wrappers and a benchmark suite** — official PettingZoo / Gymnasium registrations and
  fixed seeds, opponents and metrics, so research results are comparable.
* **Opponent modelling for search** — infer opponents' likely hands and plans from what they played
  and passed instead of sampling hidden cards uniformly; the most direct route to stronger search
  and expert iteration.
* **Smaller, faster models** — distilled and quantized networks for phones, the browser (item 8)
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
  the post-game analysis.
