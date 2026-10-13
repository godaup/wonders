# Armada integration plan

Status: plan (October 2026). `armada` is registered in `wonders/ruleset.py` with `implemented=False`;
there is no `data/armada.json` yet. This document lays out how to mechanize the expansion in slices
that each keep the full gate (`make test`) green and keep base, Leaders, Cities and Wonder Pack games
bit-identical (`tests/test_determinism.py`).

## 0. Rules source first

Every rule below is an outline from secondary sources and memory. As with Leaders and Cities, the
official rulebook decides. Sources to work from:

* Official English rulebook (Repos / Asmodee CDN):
  <https://cdn.svc.asmodee.net/production-rprod/storage/downloads/games/7wonders-armada/en/7ar-en02-rules-1598904236sa0W2.pdf>
* First English print of the rules (mirror): <https://www.tesera.ru/images/items/1320719/7AR-EN01_RULES.pdf>
* Rules discussion on naval tokens: <https://boardgamegeek.com/thread/2167013/7-wonders-armada-victory-and-defeat-token>

Outline as far as it is known (to be checked):

* Components: 7 shipyard boards, 28 flotilla figures (4 per player), 40 naval conflict tokens,
  8 extra military tokens, 24 Armada cards, 27 island cards; a team variant for 4 or 6 players.
* Each player gets a shipyard board with four fleets (red, yellow, blue, green), each on a track with a
  start space and six upgrade spaces. All boards give the same rewards per space; the naval
  construction costs differ per board.
* Building a card of a fleet's colour lets the player also pay the next space's naval construction
  cost to advance that fleet. Red gives naval strength, yellow commerce, blue victory points, green
  exploration of islands.
* The base game's military conflicts become *ground* conflicts (shields become ground shields).
  At the end of each Age, after the ground conflicts, a naval conflict compares the naval strength
  of all players. Naval victory tokens are worth 1, 3, 5 or 7, naval defeat tokens −1, −2 or −3.
* Armada cards join the Age decks and count as ordinary building cards.

**Open questions** the rulebook must settle before the engine work (each becomes a test):

1. Which actions allow naval construction: only building a card of the fleet's colour, or also a
   wonder stage, a free build (Olympia A, Halicarnassus), a leader? At most one advance per build?
2. Is the naval cost paid together with the card (one purchase, each production slot used once,
   neighbours' prices apply) or separately?
3. Exact reward of every track space per colour (naval shields, ground shields, coins, VP, island
   level, free moves, trade discounts).
4. Naval conflict: who gets which token per Age (only the strongest and the weakest? the second
   strongest?), how ties split, and what the 1-point victory token is for.
5. Whether Cities' diplomacy tokens also skip the naval conflict, and whether Armada has its own
   naval diplomacy.
6. Exploration: how many island cards are revealed per level, which one is kept, where the others go
   (bottom of the deck or out of the game), and who sees them.
7. Armada cards: copies per player count (numerals like the base deck, or a Cities-style draw),
   new symbols (naval shields on cards, naval taxes, free fleet moves) and their exact effects.
8. How shipyards are assigned (random, or tied to the wonder) and the starting position.
9. Scoring: do naval tokens go on the military row of the official scorecard or a row of their own?
   Where do island VP go?
10. Interactions with Leaders (Nero, Tomyris, Hatshepsut, Xenophon style discounts) and Cities
    (plunder, debt) as given in the rulebook's compatibility section.

**Deliverable of step 0:** `docs/armada.md`, the rules as implemented with page references, and
`wonders/cards/data/armada.json`, the functional data (costs, effects, counts; names paraphrased as
for Leaders and Cities). No rulebook PDF goes into the repository.

The rulebook hosts are unreachable from the cloud container this plan was written in (proxy denied),
so step 0 needs someone with the PDF, or the hosts added to the environment's network allow list.

## 1. Data model and content (`wonders/cards/`)

`armada.json` gains three new top-level sections next to `cards`:

```json
{
  "colors": [{"id": "island", "label": "Island", "fill": "#…", "score": "naval", "tradeable": false}],
  "shipyards": [{"name": "…", "tracks": {"red": [{"cost": {…}, "effects": [ … ]}, …6 spaces], "yellow": […], …}}],
  "islands": {"1": [ …cards… ], "2": [ … ], "3": [ … ]},
  "rules": {"naval_victory_age_1": …, "naval_defeat_age_1": …}
}
```

* **Shipyards**: a new `Shipyard` / `FleetStep` dataclass pair in `cards/wonder.py` (a step is a
  `Cost` plus `effects`, like `WonderStage`). The rewards are ordinary effect specs, so they reuse
  `EFFECT_BUILDERS`.
* **Islands**: `Card`s of a new data-defined colour (`Color.from_spec` via `"colors"`, the
  mechanism Rinascita already uses), drawn only by exploration and kept out of `build_age_deck` the
  way white cards are (`Content.islands(level)`).
* **Armada cards**: ordinary cards in `cards`. If they use `at_players` numerals nothing changes in
  `build_age_deck`; a Cities-style draw needs a branch in both `build_age_deck` and
  `decks.deck_size`. Add `'armada'` to `decks.DECK_EXPANSIONS` so `check_decks` covers every
  combination with Leaders and Cities at 3 to 7 players.
* **New effect types** (each with an `EFFECT_SCHEMAS` entry, `tests/test_builder.py` enforces it):
  `naval_shields`, `advance_fleet` (free move, fixed colour or a choice), `explore` (island level),
  and whatever the Armada cards need (naval taxes etc.).
* **New meta rules** in `ruleset.RULES`: naval victory per Age and naval defeat per Age, so extension
  authors can tune them like the ground tokens.

Acceptance: `load_content('base', 'armada')` loads, `check_decks` passes for all combinations, the
gallery renders the Armada cards, islands and shipyards.

## 2. Engine (`wonders/engine/`)

All new state defaults to inert values, so games without Armada take exactly the old code path.

**Player** (`player.py`)

* `shipyard: Optional[Shipyard]`, `fleets: Dict[Color, int]` (position per colour, 0 = start),
  `naval_tokens: List[int]`, `islands: List[Card]`, `pending_explores: List[int]` (levels),
  `pending_fleet_moves: int`.
* One mutator `advance_fleet(color)` that bumps `Player.version` and clears `_cache` (fleet rewards
  feed production, shields and trade rules), like `add_card` / `add_stage`. Never mutate `fleets`
  directly.
* New effect hook `naval_shields()` on `Effect` (default 0) and an aggregate `Player.naval_shields()`.
  Fleet steps reached and islands kept count as built effects, so every existing hook (production,
  `trade_rules`, `victory_points`, …) works for them unchanged.

**Naval construction: a follow-up decision before resolution.** The choice to advance a fleet is a
real strategic choice with a cost, so it must be visible to agents and the network. Three options
were weighed:

* a `compare=False` field on `Action` like `payment` / `stage`: invisible to the action space, so no
  learned agent could ever choose it;
* a fifth `ActionKind`: changes `ActionSpace` (index `card * 4 + kind`) and breaks every checkpoint;
* **a new `DecisionKind.NAVAL`** (recommended): in `age_steps`, after the `PLAY` actions arrive and
  before `_resolve`, every seat whose action qualifies (open question 1) and who can afford the
  *combined* cost of card plus next step is asked at once. Options are the card being built;
  `Action.build(card)` advances, `Action.discard(card)` declines (the card is still built; the
  existing index space covers both). The combined cost is priced in one call of
  `trade.cheapest_payment` over `card.cost + step.cost`, which keeps the "each slot once per build"
  rule; `payment_options` gets the same combined variant so the GUI can offer splits.

Asking before resolution keeps the turn simultaneous and the payment exact. The record stores it
generically in `extras` (`[turn, 'naval', seat, *move]`), replay matches it by `entry[:3]`.

**Fleet rewards and powers** (in `_power_steps`, after every turn and during recruitment):

* `EXPLORE` (single seat): draw the island cards for the level from `Game.island_decks[level]`,
  options = the drawn cards, `Action.build(island)` keeps one; the rest go where the rulebook says.
* `FLEET_MOVE` (single seat, only if a reward lets the player pick the colour): options are four
  synthetic fleet cards (`Red fleet`, …) so the decision fits the `Card`-based `DecisionPoint`.

**Naval conflict**: `Game.resolve_naval(age)` called right after `resolve_conflict(age)` in
`age_steps`, a no-op when no player has a shipyard. Token values from `Game.rules`; tokens go to
`Player.naval_tokens` so `sum(tokens)` stays the ground military score. Nero / Tomyris style hooks
apply only where the rulebook says so.

**Scoring**: a `naval` column in `ScoreBreakdown` (naval tokens, plus island VP if the rulebook puts
them there). Because `as_dict()` feeds the golden digest, `naval` is left out of `as_dict()` unless
the game uses Armada; the digest then stays unchanged. If open question 9 says the official scorecard
adds naval tokens on the military row, keep the column anyway for analysis and sum it into the
displayed military row only in the renderer.

**Setup**: `Game.__init__` assigns shipyards (random or per the rulebook) and shuffles the island
decks from the game's RNG *after* every existing draw, so the RNG stream of non-Armada games is
untouched.

**Record** (`record.py`): setup gains `shipyards` (per seat) and `island_decks` (ordered, like
`leader_box`), both omitted when empty so old records and their JSON stay byte-identical.
`replay()`, `replay_observations()` and `branch()` need no other change.

Acceptance: one rules test per open question (as answered), replay round trip of random Armada games
at 3 to 7 players with every expansion combination, golden digest unchanged.

## 3. Observation, worlds and the env

* **`CityView`**: `shipyard` (name), `fleets` (position per colour), `naval_shields`,
  `naval_tokens`, `islands` (kept island cards are public), `next_fleet_costs` (per colour, like
  `next_stage_cost`).
* **`Observation`**: `naval_options` during a `NAVAL` decision (combined cost and splits), the drawn
  islands during `EXPLORE`. Islands the explorer saw and declined go into the explorer's
  `seen_cards` only; the island decks' order is hidden from everyone.
* **History**: a `naval` move is public (the fleet visibly advances); declined islands are hidden
  from opponents. Extend `tests/test_table_encoder.py`'s check that hidden state never changes
  another seat's encoding to island decks and declined islands.
* **`worlds.sample_world`**: copy fleets, naval tokens and islands; resample the island decks from
  the cards not publicly accounted for (excluding what the viewer has seen leave the deck).
* **Env**: `WondersEnv` handles new decision kinds generically (`_record_turn`); the `dense` reward
  picks up naval tokens through `score()`. Add `naval` to `agents.bench.CATEGORIES`.

## 4. Agents

* **`Agent` base**: defaults for the new kinds, so every scripted agent plays Armada at once: advance
  when the step's reward is worth its coin cost (cost proxy), keep the island with the best
  `strength.rate_card` value, pick the fleet furthest behind its reward.
* **`HeuristicAgent`**: features for naval strength (expected naval token value against *all*
  players, Age scaled like the ground military term), fleet step rewards by kind, exploration value
  by level; weights tuned with `bench.py` as before.
* **`BoardAwareAgent`**: read the shipyard's next costs and plan production for them, as it already
  does for unbuilt wonder stages.
* **Archetypes**: the militarist values the red fleet, the economist the yellow, the civic the blue,
  the scientist the green.
* **Strength tools**: extend `wonder_values` / `strength.rate_wonder` to shipyards (boards rotated with
  common random numbers), and rate the Armada cards and islands for `docs/card-strength.md`.

Acceptance: a bench run (`wonders bench -e armada`) where the heuristic family beats random and
greedy with Wilson intervals not overlapping, and a short report in `docs/armada.md`.

## 5. Learned agent

* **Encoder `table-v3`** (new `Vocab.encoder_kind`; `table-v2` stays for existing checkpoints):
  fleet positions and next costs per city, naval shields and tokens, kept islands, an Armada
  expansion flag, `naval` in `SCORE_KEYS`, and the new decision kinds in the context block. Changing
  `table-v2` itself would break the shapes of the bundled model.
* **Existing checkpoints in Armada games**: `NeuralAgent` routes decision kinds the checkpoint does not
  know to the scripted base defaults (the pattern of `leader_policy=False`), and the GUI marks the
  neural opponent as not trained for Armada rather than refusing the game.
* **Training**: imitation from `HeuristicAgent` with Armada in the expansion mix, then PPO, compared
  with `wonders train h2h` against `ppo_universal_v2`; report in `sweeps/ARMADA.md`.

## 6. Play, GUI, i18n and docs

* `HumanSession`: show the shipyard under the wonder board, a "also advance the fleet" choice after a
  qualifying build (with the combined cost and payment splits), the island pick, the naval conflict
  in the end-of-Age summary and a `naval` score row.
* German translations of every new card, island, shipyard and GUI string (`tests/test_i18n.py`
  fails otherwise); placeholder checks apply.
* Renderer and gallery: shipyard boards and island cards.
* Expansion builder: shipyard and island forms can follow later; the first version only needs
  `check_project` to accept files with `shipyards` / `islands`.
* Flip `implemented=True`, update the three tests that pin the flag (`tests/test_expansions.py`,
  `tests/test_play.py` twice), the README expansion table, `CLAUDE.md`, `ROADMAP.md` and
  `CHANGELOG.md`. Bump `RULES_VERSION` only if a base rule changes (none is planned).

## Out of scope

* The team variant (4 or 6 players): the engine has no teams; it would be a separate ruleset option.
* Armada for the 2nd edition (2020): different card set, out of scope like the 2nd edition itself.

## Slices (one PR each)

| # | Slice | Depends on | Keeps digest |
|---|---|---|---|
| 0 | `docs/armada.md` rules transcription, open questions answered | rulebook | yes |
| 1 | `armada.json`, shipyard / island / effect parsing, deck checks, gallery | 0 | yes |
| 2 | Fleets, `NAVAL` decision with combined payment, naval conflict, scoring, records | 1 | yes |
| 3 | Islands, `EXPLORE` / `FLEET_MOVE`, Armada card effects | 2 | yes |
| 4 | Observation, worlds, env, base agent defaults, partial-observability tests | 3 | yes |
| 5 | Heuristic and board agents, bench report, `implemented=True`, GUI and i18n | 4 | yes |
| 6 | `table-v3` encoder, NeuralAgent fallback, training run, strength reports | 5 | yes |

Slices 1 to 4 keep `implemented=False`, so the GUI keeps the expansion greyed out until the game is
complete and playable by every agent.
