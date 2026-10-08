# Roadmap

Planned work, roughly in priority order. Each item says why it matters, what is already in place,
and what is left. Finished work moves to `CHANGELOG.md`; training reports live in `sweeps/`.

## 1. Retrain after the October 2026 rules fixes

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

**Status: in progress** (branch `search-agent`, not merged).

* `wonders/agents/worlds.py` rebuilds a playable game from one seat's observation and samples the
  hidden cards (opponents' hands, hidden discards / burials, the guild / black-card draws,
  opponents' leaders), so search cannot use hidden information.
* `wonders/ml/search.py` (`SearchAgent`) plays the network's top candidates out in shared sampled
  worlds for a few turns, then uses the value head.
* Old-rules results against plain `s5-anneal` at the same tables (800 games each): 3 turns ahead
  with 16 worlds ranks above it in 60.2% of games; 2 turns / 32 worlds 58.0%; full rollouts are
  worse than no search (too noisy). Gains seem to level off around 56–60%.

**Next.**

1. Finish the depth / width sweep, re-run the best setting under the new rules, merge.
2. Expert iteration: record the search agent's choices (and values) in self-play, distil them into
   the network (imitation on the search policy), continue PPO, repeat while head-to-head improves.
3. Alternatives if flat search stalls: information-set MCTS or decoupled UCT for the
   simultaneous moves (alpha-beta does not fit: 3–7 players, simultaneous moves, hidden cards).

## 3. Post-game analysis

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

## 4. Remaining expansions

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

### Babel (registered, `implemented=False`)

Two independent modules:

* **Tower of Babel** — law tiles drafted at the start and placed on a shared tower during the game;
  the top three active tiles change rules for everyone (costs, trade, military, science, …).
  Engine: a tile draft, a "place a tile instead of playing a card" action, rule modifiers as
  global effects queried through the existing hooks.
* **Great Projects** — a shared project card per Age that players join by building cards of its
  colour (paying a participation token); completing it rewards participants, failing it penalises
  them. Engine: per-Age project state, participation payments, end-of-Age resolution.

### Wonder Pack (not yet registered)

Four extra wonder boards (Great Wall, Stonehenge, Manneken Pis, Abu Simbel) with unusual stage
rules (e.g. stages built in any order, copying a neighbour's board, burying leaders). Mostly data
plus a few board powers; a natural first step because it reuses the existing turn structure.

### Out of scope for now

* **7 Wonders, 2nd edition** (2020) — different card set, Edifice expansion and some rule changes;
  it would be a separate ruleset rather than an expansion of this one.
* **7 Wonders Duel** — a different two-player game.

## Smaller items

* Publish to PyPI once the API settles (release flow in `CLAUDE.md`).
* A replay viewer in the GUI for recorded games (`GameRecord` JSON), reusing the review screen of
  item 3.
