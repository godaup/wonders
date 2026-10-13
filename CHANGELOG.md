# Changelog

## Unreleased

* GUI fixes: the menu's "Your wonder" list offered only "Random" (the boards were never added to it);
  the debt and diplomacy counters now show whenever the game's cards or boards can hand out those tokens
  (`session.token_kinds`: Cities, Rinascita, the Wonder Pack's diplomacy stage), not only with Cities.
* Card texts: effects without a description no longer print their class name (science set bonus and
  science copy get their own text, others a readable fallback); Rinascita 0.1.1 describes its bank,
  science, set and defeat effects like the official cards.
* Tests pin down military resolution: every pair of neighbours around the ring fights once, a diplomat
  leaves the ring so its neighbours fight each other, and three players with a diplomat fight once.
* **Community expansions catalog**: the data extensions in `extensions/` ship with the package
  (`wonders/catalog/`, copied by the new `setup.py` build hook; a checkout reads `extensions/`). The GUI
  has a "Community expansions" page (header and menu) to search them by name, author, card or board and
  switch them on or off; on the command line `wonders ext catalog [words]`, `wonders ext enable <id>`
  and `wonders ext disable <id>`. Switched-on ids live in `$WONDERS_HOME/extensions.json` and load at
  every start; saved games and records that used a catalog expansion load it on demand
  (`extensions.provide`). Your own extensions folder takes precedence over the catalog.
* **Rinascita**, the project's first community expansion (`extensions/rinascita/`, data only): from
  plague to enlightenment in three Ages (a harsh Age I of debt and plunder, Patronage with the new Arts
  colour, Enlightenment science), one card per player per Age, two guilds and three wonder boards, with
  German names; tuned with the strength tools (balance table in its README).
* Fix: `wonders ext check` warned that an extension locale's colour labels are never shown (they are,
  on the cards and in the menu).
* Deck checks (`wonders.cards.decks`): every Age deck must split evenly into the hands at 3 to 7 players,
  alone and with Leaders and Cities, with the expansion's `extra_guilds` rule applied as the game applies
  it. Leftover cards are now an error in the expansion builder (the expansion is not loaded) and in
  `wonders ext check`; a changed hand size or too small a guild pool is a warning. The builder shows the
  Age III deck sizes next to the extra guilds rule. Fixed: `wonders ext check` ignored an extension's meta
  rules when sizing decks. `wonders ext new` scaffolds a card with one copy per player.
* Card and board strength in the expansion builder (`wonders.agents.strength`, `wonders.play.strength`,
  `docs/card-strength.md`): a rate card values a card or wonder side in VP while it is edited (effects,
  costs, chains and powers measured in reference positions, compared with the official cards of the same
  Age and colour); after saving, changed cards and boards are simulated in the background (paired runs
  from the same position); "Simulate the whole expansion" plays them together and recalibrates the rate
  card. The overview lists every card and side with the best estimate. Calibrated on simulated values of
  the official cards (`python -m wonders.agents.strength`).
* Strength of the wonder board sides (`wonders.agents.wonder_values`, `python -m wonders.agents.wonder_values`):
  random deals played with the board assignment rotated over the same decks, regressed on the sides
  at the table (own side and opponents'), with a value per side and a partially pooled deviation per
  expansion set and player count, cluster-robust standard errors; results in `docs/wonder-strength.md`.
* The Wonder Pack (expansion `wonderpack`): the Great Wall (stages in any order), Abu Simbel (entomb a
  leader for twice its cost; needs Leaders), Stonehenge (stone symbols; side B scores the buried card's
  colour at the end) and the Manneken Pis (copies its neighbours' stages; +4 coins), every power
  mechanized from the official rules sheet. A WONDER action may choose its stage (`Action.staged`,
  `Game.stage_options`, stage pills in the GUI; stored in records); the board renderer draws stages built
  out of order and copied stages; the GUI menu offers Abu Simbel only with Leaders. Discard builds owed
  in one turn resolve in the FAQ's order (Halicarnassus, Great Wall, Manneken Pis, Solomon, Courtesans).
  Without Cities, plunder no longer creates debt (no debt tokens in play).
* New card colours for custom expansions: an expansion file declares them under `"colors"` (label,
  fill, scorecard row, whether neighbours may buy their production) and cards use them like the
  official ones; the card renderer, the GUI and `wonders ext validate` support them. `Color` is now
  an open class instead of an `Enum` (same constants and behaviour).
* Babel is removed from the roadmap and from the list of registered expansions.
* Expansion builder in the GUI ("Expansion builder" in the header, `#builder`): create and edit an
  extension's card colours, cards and wonder boards (both sides, stages, board powers) in forms with
  live card and board previews, change its meta rules, see problems on the items they belong to, save
  it into the extensions home (the game menu offers it right away), play it, run a balance check with
  scripted agents against the base game, and download it as a zip.
* Meta rules: expansions may change starting coins, coins for a discard, coins per treasury point,
  the science set bonus, military token values and the number of extra guilds (`"rules"` in an
  expansion file; `wonders.ruleset.RULES`), setting a value or changing it by a delta (`{"add": n}`).
  Rules apply in a fixed order whatever order the expansions were chosen in (base, bundled, then
  extensions by id): deltas add up, a set value replaces what came before, the result is clamped to the
  rule's range (before, starting coins took the maximum; the bundled expansions are unaffected).
* Effect schemas (`wonders.cards.schema`): a parameter description for every effect type;
  `Registry.register_effect(..., schema=...)` lets plugins describe theirs.
* Extension folders are built when they load; a folder whose cards or wonders do not build is skipped
  with a warning instead of being registered half-way. Plugins now load before folders, so folders can
  use their effect types.
* Contribution checks (`wonders.contrib`, `CONTRIBUTING.md`): data extensions go to `extensions/<id>/`,
  plugin packages to `plugins/<name>/` (with a complete example plugin), translations of the game to
  `wonders/i18n/<lang>.json`. `wonders ext check` validates any of them (manifest metadata, data-only
  files and size limits, clashes, deck sizes, games played and replayed with every expansion, agent
  and model, translation placeholders and coverage); `wonders ext scorecard` plays a model against
  the bundled network (ranked-above rate with interval, per combination, player count and score
  category). GitHub Actions run them on pull requests with the base branch as referee and post the
  report (`contributions.yml`, `contributions-comment.yml`); merged models are measured again for
  `LEADERBOARD.md` (`leaderboard.yml`). `WONDERS_EXTENSIONS` takes several folders; `wonders ext new
  --author`.
* Fix: a plugin's `MANIFEST` is now read from the module of its `register` function, as documented
  (it was ignored).
* Python 3.11–3.14 are supported (3.14 added; 3.10, end of life in October 2026, dropped).
* Local storage (`wonders.storage`, `$WONDERS_HOME`, default `~/.wonders`): the GUI saves every game
  after each move and can resume or re-open it (`HumanSession.saved` / `restore`: the seed plus the
  logged decisions); `wonders gui --no-save` turns it off. New GUI pages: "Past games" with the
  player's Elo rating and its graph (agents have fixed ratings fitted on 20,000 simulated games,
  `wonders.play.rating`; `python -m wonders.play.rating` refits them), and "Achievements"
  (`wonders.play.achievements`): 42 feats and career goals in four tiers (bronze to legendary, worth
  points) — general, Leaders, Cities, opponent-specific and career (table sizes, wonders, Elo goals),
  calibrated on 1,500 simulated games. Ratings and achievements are derived from the saved games, so
  new ones count old games too.
* Agents have display names (`AGENT_NAMES`: Fortuna, Icarus, Odysseus, Hephaestus, Metis, Eris,
  Pythia, …) in the GUI, as default seat names and as CLI aliases (`-a pythia,metis`); the ids stay.
  Extensions now default to `$WONDERS_HOME/extensions`.
* German translation (`wonders/i18n/`): the GUI, cards, wonder boards, tooltips and the post-game
  analysis; language picker (browser language by default, `?lang=de`), `wonders gallery --lang de`.
  Further languages are data (`docs/extensions.md#translations`).
* Extensions (`wonders.extensions`, `docs/extensions.md`): expansions, models and locales from data
  folders; card effects and agents from pip plugins; `wonders ext list | validate | new`; game
  records pin extension versions and hashes.
* Post-game analysis: `wonders analyse` and "Analyse my moves" in the GUI rate every move with
  network-guided search (accuracy, key moments with reasons, expected-margin graph), find patterns
  across the game (trade, military by one shield, science, wonder, coins, passed cards, discards) and
  replay key moments with the better move (`GameRecord.branch`); ratings and charts per aspect
  (category ranks vs the table, points lost per move, accuracy by Age and move type, city colours,
  military per neighbour, economy, wonder).
* The bundled network is retrained under the October 2026 rules (`r1-a`, `sweeps/RETRAIN.md`):
  52.2% head-to-head over the previous network; leader values re-estimated.
* Rules fix: cards used for a wonder stage stay under the board and discarded leaders return
  to the box (Leaders rulebook) — neither goes to the discard pile, so Halicarnassus / Solomon
  can only build discarded Age cards.
* Rules fix: neighbours may only buy the board's resource and brown / grey card production;
  yellow cards (Forum, Caravansery, ...), wonder stages and leaders produce for their owner only
  (`Player.tradeable_slots`, `CityView.trade_production`).
* GUI: card effect text wraps; larger wonder board and every board's stages with their effects;
  resources, VP labels and hover tooltips for every city, card and stage; leaders shown as
  cards; a passing-direction banner; duplicate cards merged; choose which neighbour to buy
  from (`Action.paying`, `Game.payment_options`, recorded in `GameRecord`).
* Builders Guild text now says it counts your and your neighbours' wonder stages.
* Complete base game: full deck and wonders, with the special wonder powers mechanized
  (Babylon B, Olympia A/B, Halicarnassus).
* Optional, combinable expansions: Leaders (official deck, draft and recruitment) and
  Cities (official deck), with every leader and black-card power mechanized and
  checked against the official rulebooks; the Leaders guilds and the Rome, Petra and
  Byzantium boards; the training env supports the Leaders draft and recruitment.
* Score breakdown gains `leaders` and `cities` columns.
* Wonder-board-aware agents (`BoardAwareAgent`, `BoardFocusAgent`); observations expose the
  whole wonder board and own production.
* Agents: baselines, strategy agents and a tuned `HeuristicAgent`; `wonders.agents.bench`
  for win rates with confidence intervals.
* Training environments: fixed `ActionSpace` with masks, `WondersEnv` (multi-agent),
  `SingleAgentEnv` (Gym-style), `SyncVectorEnv` / `SubprocVectorEnv`, `CardEncoder`.
* Replayable `GameRecord` notation (version 2 adds power-granted decisions).
* About 2.8x faster engine and observations, behaviour verified unchanged by a golden test.
* CLI: `wonders play | replay | bench | agents | gallery | table`.
* Optional learned agent (`pip install wonders[ml]`): policy/value network, imitation
  learning, PPO and `wonders train imitate | ppo`.
* Bundled universal network (`ppo_universal_v2.pt`) for every expansion combination and 3–7
  players, replacing the base-game-only model; the `table-v2` observation encoder (every city
  seat-relative, rich hand / private / history blocks) is the default for new models. The bundled
  network is the learning-rate-decayed `s5-anneal` (53.2% head-to-head over its predecessor).
* Training: self-play league with a tiered opponent mix (scripted / recent self / anchors),
  long-lived self-anchors, large neural opponents on the GPU, linear learning-rate decay, a
  rank-shaped reward option, resumable JSON sweeps (`wonders train sweep`).
* Evaluation: `wonders train eval | h2h | exploit | leaders` — scripted lineups, models
  head-to-head with random seating, exploitability, Leaders diagnostics.
* Agents: data-driven leader play for the heuristic family (`leader_values`), the adaptive
  scripted agents `adaptive` and `rival`, leader-decision ablations; `NeuralAgent` plays the
  Leaders draft and recruitment with its own policy.
* `wonders gui`: play against agents in the browser (local only).
* `Effect.counts()` hook describing what counting effects pay per unit.
* Requires Python 3.10+.

## 0.1.0 (2022-01-21)

* First release on PyPI.
