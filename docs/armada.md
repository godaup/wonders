# Armada: implementation plan

Status: **plan** (the expansion is registered in `wonders/ruleset.py` with `implemented=False`).
The rules below come from the official English rulebook (Repos Production, 2018,
`7ar-en02-rules`). Values marked *(fig.)* were read off the rulebook's board illustration and
must be checked against the physical board or the "Description of New Effects" sheets before
they become data. Rule questions the rulebook leaves open are collected in
[Open questions](#open-questions); check them against the official FAQ (www.7wonders.net/faq),
as was done for Leaders and Cities.

## 1. What Armada adds

| Component | Contents | Where it lands |
|---|---|---|
| 7 Shipyard boards | 4 naval tracks (red, yellow, blue, green) × 7 spaces (start + 6). Space effects are the same on every board; only the construction costs and the track with the **Wonder symbol** differ. | `armada.json` `"shipyards"` + shared `"tracks"` |
| 24 Armada Age cards | 8 per Age; 5 are marked 4+ (removed at 3 players). Each Age, **N random cards** are added to the base deck. | `armada.json` `"cards"` |
| 27 Island cards | 3 levels × 9, separate decks, different back. Island resources cannot be bought by neighbours. | `armada.json` `"islands"` |
| Siracusa | A new wonder board (A/B), only used with Armada. | `armada.json` `"wonders"` |
| Tokens | Naval victory tokens (1/3/5/7), naval defeat tokens (−1/−2/−3), 2 boarding tokens, extra military tokens. | `Player` state, `ScoreBreakdown` |

### Rule changes (rulebook § Gameplay A–E, scoring)

* **A. Extra turn**: hands of **8** cards (9 with Cities). The deck gets N Armada cards per Age,
  as Cities does with black cards.
* **B. Naval Construction**: building a red, yellow, blue or green **card** may also advance
  that colour's fleet one space. Building a **wonder stage** may advance the fleet with the
  Wonder symbol. The card (or stage) and the Naval Construction are **paid together** in one
  payment (resources, trade and coins), but their effects apply in order: card or stage first,
  then the fleet. This is allowed on chained and Olympia free builds (the naval part is still
  paid). It is **never** allowed on builds from the discard pile (Halicarnassus, Solomon).
* **Shipyard tracks** *(fig.)*:

  | Space | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | Red: naval strength (value at the current space) | 1 | 2 | 3 | 4 | 5 | 8 |
  | Yellow: commercial level (current) | 1 | 1 | 2 | 2 | 3 | 4 |
  | Yellow: on reaching the space | – | 2 coins, tax 1 | 3 coins | 4 coins, tax 2 | 5 coins | 2 naval strength, tax 4 |
  | Blue: VP (current) | 1 | 2 | 4 | 6 | 8 | 10 (+1 naval strength) |
  | Green: on reaching the space | – | explore L1 | – | explore L2 | – | explore L3 |

  "Current" values are **positional**: they are not summed over the spaces passed. "On
  reaching" values fire once, when the fleet arrives.
* **Taxes**: are applied **at the end of the turn**. Every *other* player loses
  `max(0, tax − own commercial level)` coins. If several taxes are raised in one turn, a player
  pays only the highest. Coin losses come last in the turn order (below).
* **Exploration**: one explorer at a level draws 4 islands. If several players explore the
  same level on the same turn, they share that level's whole deck equally (the remainder is not
  dealt). Each keeps 1 island and the rest are reshuffled into the deck. Island levels are
  independent of the Age.
* **C. Naval conflict**: at the end of each Age, after the military conflicts, the engine
  compares naval strength across **all** players. Naval strength comes from cards, the
  shipyard, islands and the wonder. Tokens are given in this order: last, first, second.

  | | Age I | Age II | Age III |
  |---|---|---|---|
  | Last | −1 | −2 | −3 |
  | First | 3 | 5 | 7 |
  | Second | 1 | 3 | 5 |

  Ties: everyone tied for last takes the defeat token. Players tied for first each take the
  **second-place** token, and nobody takes second. Players tied for second take nothing. If
  every player has the same strength, nobody takes a token.
* **D. Boarding**: some red cards give a boarding token to a designated player (the symbols
  point two seats away). At the end of the Age, that pair also fights an extra, normal military
  conflict. Diplomacy tokens do not stop it, and a player can be boarded twice.
* **E. Selling** (discarding): gain 3 coins **or** advance the yellow fleet one space for
  free.
* **Turn resolution order**: (1) pay all costs, (2) advance fleets, (3) card effects except
  coin losses, (4) shipyard effects except coin losses, (5) builds from the discard pile (no
  Naval Construction), (6) coin losses.
* **Scoring**: each **purple card is capped at 10 VP**. Naval tokens and islands get their own
  score rows. Blue fleet VP goes on the card row. Science already scores `n²` for more than 3
  identical symbols (`science_score`), so nothing changes there.
* **With Cities**: hands are 9 cards. A tax or coin loss a player cannot (or does not want to)
  pay becomes **debt**, one token per coin.
* **Team variant** (4/6 players): out of scope (see [Later](#later)).

## 2. Design

The plan follows the repo's design rules: content is data, mechanics are `Effect` hooks, turn
structure lives in the engine coroutines, and everything stays inert when `armada` is not in
the ruleset. The golden digest in `tests/test_determinism.py` must not change for non-Armada
games.

### 2.1 Data (`wonders/cards/data/armada.json`)

```jsonc
{
  "colors": [{"id": "island", "label": "Island", "score": "islands", "tradeable": false}],
  "tracks": {               // shared by every shipyard: 6 spaces per colour (index 1..6)
    "red":    [[{"type": "naval_shields", "n": 1}], ...],
    "yellow": [[{"type": "commercial_level", "n": 1}], [{"type": "commercial_level", "n": 1},
               {"type": "coins", "n": 2}, {"type": "tax", "n": 1}], ...],
    "blue":   [[{"type": "victory_points", "n": 1}], ...],
    "green":  [[], [{"type": "explore", "level": 1}], ...]
  },
  "shipyards": [{"name": "...", "wonder_track": "red",
                 "costs": {"red": ["W", "2$", ...], "yellow": [...], ...}}],   // 6 costs per track
  "cards": [...],           // 24 Age cards, "deck": "draw" (N random per Age, like black cards)
  "islands": [{"name": "...", "level": 1, "effects": [...]}],
  "wonders": [{"name": "Siracusa", ...}]
}
```

* **Positional vs on-reach** space effects: a space's effects are *status* effects (naval
  shields, commercial level, VP) that hold only while the fleet is on that space. The engine
  exposes the hooks of the current space only. *Trigger* effects (coins, tax, explore) fire
  once through `_activate` when the fleet arrives. The engine does not need to know which is
  which: status hooks are read from the current space, and `coins_on_build` / `on_build` /
  the new trigger hooks run on arrival. This is the same uniform-hook rule cards follow.
* **The Armada deck draw** generalises the Cities special case in `content.build_age_deck`
  (`black` → N random cards). Add a per-expansion `draw_per_age` (or a card-level `"deck":
  "draw"`), so both Cities and Armada use one code path. `decks.check_decks` must then accept
  hands of 8 (Armada), 8 (Cities) and 9 (both), and the 3-player removal of 4+ cards
  (`at_players`) applies before the draw.
* **Islands** are `Card`s of a new non-official colour `island` with `tradeable=False` (this
  already gives "neighbours cannot buy island resources" through `Color.tradeable`). They are
  kept in `Player.islands`, not in the Age decks, and count as neither guild colours nor
  colour counts.
* **Shipyard assignment**: a random board per seat at setup, stored in the `GameRecord`
  (`shipyards`) so replays are exact. Siracusa joins the wonder pool only when `armada` is
  active, as Rome does with `leaders`.

### 2.2 New effect hooks (`wonders/effects/base.py`, neutral defaults)

| Hook | Used by |
|---|---|
| `naval_shields() -> int` | red track, red Armada cards, islands, Siracusa |
| `commercial_level() -> int` | yellow track (status) |
| `tax() -> int` | yellow track trigger spaces |
| `explores() -> Tuple[int, ...]` | green track, the island-draw cards (top-card draw with no choice: separate `island_draws()`) |
| `fleet_moves() -> Tuple[track \| 'any', ...]` | cards that advance a ship for free, Siracusa A (one ship of choice), Siracusa B 2 (two *different* ships) |
| `boards() -> Tuple[int, ...]` | boarding red cards (seat offsets, e.g. +2 / −2) |
| `coin_losses(ctx)` through `on_build` | pirates: others lose coins equal to their commercial level (the existing `plunder_per` with `what='commercial_level'`) |
| `vp_cap` | not a hook: the purple cap is a meta rule, `rules.guild_vp_cap` (default `None`, Armada 10) |

`Tally` gets the metrics `commercial_level` (Customs Office: 2 VP per level), `naval_tokens`
and `fleet_position(track)`, if the card texts need them.

**One coin-loss path.** Add `Game._lose_coins(i, amount, kind)`. It subtracts the commercial
level for `kind='tax'` (and for other losses if the FAQ says so), turns what cannot be paid
into debt when `Context.debt_tokens` (Cities) is set, and is called only while
`_flush_pending` runs. That flush already applies "gains, then losses" at the end of the turn,
which is step 6 of the resolution order. Taxes are collected per turn and per victim, and only
the highest is applied.

### 2.3 Engine (`wonders/engine/`)

* **`Player`**: `shipyard`, `fleets: Dict[track, int]`, `islands`, `naval_tokens`,
  `boarded_by: List[int]`. Add `advance_fleet(track)`, which clears `_cache` and bumps
  `version` the way `add_card` / `add_stage` do (cache invariant). Add aggregates
  `naval_strength()` and `commercial_level()`, and fold the current space's effects into
  `production()` / `victory_points` in the same way as built effects.
* **The action carries the naval choice.** Add `Action.naval: bool`, which is **part of
  equality**, unlike `payment` / `stage`, because it changes what happens and what it costs.
  `legal_actions` emits the naval variant for BUILD / WONDER / FREE when the combined cost is
  affordable and the fleet is not at the top. For DISCARD, `naval=True` means "advance yellow
  instead of +3 coins".
* **Combined payment**: `_card_payment` / `_stage_payment` get the naval cost added:
  `Cost(coins, resources) + track cost` goes to `cheapest_payment` in one call (chained or free
  builds pay only the naval part). `payment_options` and the memo key `Game._payments` must
  include the naval flag.
* **Apply order** in `_do_build` / `_do_wonder` / `_do_free`: settle the combined payment, add
  the card and `_activate` it, **then** `advance_fleet` and `_activate` the space it reached.
  Coins are immediate, while taxes, explorations and coin losses go to the pending queue or to
  power steps. `_do_free` from the discard pile rejects `naval=True`.
* **New decision kinds** in `_power_steps`, resolved after card effects and before discard
  builds (resolution order 4–5):
  * `EXPLORE`: the island options for each exploring seat. The engine groups explorers by
    level and either deals 4 cards or splits the whole level deck. Several seats can decide
    at once, like `PLAY`. A pick is `Action.build(island)`, the same convention as DRAFT.
  * `ADVANCE_FLEET`: a choice of track (Siracusa, and any card that says "a ship of your
    choice"). The four tracks are four **virtual cards** (`Red Fleet` … `Green Fleet`) in the
    card vocabulary, so the pick is `Action.build(card)` and the action space keeps its shape.
* **Conflicts**: `resolve_conflict(age)` adds (a) a military conflict for every boarding pair
  (Tomyris and Nero still apply; diplomacy is ignored), then clears the boarding tokens, and
  (b) `_resolve_naval(age)` with the rank and tie rules above, using
  `rules.naval_victory(age)` / `naval_second(age)` / `naval_defeat(age)` as new meta rules
  (base values `None`, set by Armada).
* **Scoring**: new `ScoreBreakdown` columns `naval` and `islands` (zero without Armada, so
  English output and existing digests do not change). Blue fleet VP goes to `civilian`, as the
  scorecard says. Apply the purple cap per card where guild VP are summed, including the
  Olympia B copies in `_copy_guilds`.
* **Hands**: the hand size comes from the deck builder. `age_steps` already plays
  `len(hand) − 1` turns, so 8- and 9-card hands need no change there.
* **Record** (`record.py`): the move entry gets a naval flag, and setup stores `shipyards`.
  Islands use **recorded offers**: each exploration stores the cards offered per seat, and
  replay deals from the record instead of a shuffled deck. This keeps the "replay without RNG"
  property, in the same style as `leader_box`. The live game shuffles the island decks from
  the game's RNG at setup and after every return. A record without Armada fields loads
  unchanged.
* **Worlds** (`agents/worlds.sample_world`): resample the island decks (unseen cards only;
  islands that were offered and returned are known to the seat that saw them, so they are
  re-mixed in) and the Armada cards drawn into future Ages.

### 2.4 Observation and encoders

* **`CityView`** (public): shipyard (name, costs, wonder track), fleet positions, islands
  (face up), naval strength, commercial level, naval tokens, boarding tokens given and
  received.
* **`Observation`** (private): the islands offered at an `EXPLORE` decision. Table-level: the
  size of each island deck and the Armada cards still possible this Age. Islands that were
  offered and returned stay hidden from the other seats (partial observability test, as in
  `tests/test_table_encoder.py`).
* **Action space**: the index becomes `card * 4 + kind` for `naval=False` and
  `n_cards * 4 + card * 4 + kind` for `naval=True`, but only when the vocabulary contains
  Armada (`Vocab.meta()` records it). Old checkpoints keep their size and never see naval
  indices, and the GUI's checkpoint/vocabulary check already refuses them for Armada tables.
* **Encoder `table-v3`** = `table-v2` + a per-seat Armada block (4 fleet positions, naval
  strength, commercial level, naval token sum, island count / production) + a context block
  (Armada active, island deck sizes). Keep `table-v2` as is: changing its size would break
  every bundled checkpoint. `DEFAULT_ENCODER` stays `table-v2` until an Armada-aware model is
  trained.

### 2.5 Agents

* `Agent` base: `explore(obs, options)` and `advance_fleet(obs, options)` routed through
  `act` like `draft_leader` / `recruit`. The default is the best island by `strength.rate_card`
  and the lowest fleet.
* `HeuristicAgent`: values the naval variant as the marginal value of the next space. Red is
  the expected change in naval tokens given the other seats' strengths (all players compete,
  so this resembles the military term but is table-wide). Blue is the VP delta. Yellow is coins
  plus the tax inflicted and the protection gained. Green is the expected island value by
  level. All of this is weighed against the naval cost in coins. Discard vs advancing yellow
  goes through the same term.
* `BoardAwareAgent`: adds the wonder-symbol track to stage planning.
* Weights get tuned with `bench.py`, and `strength.py` / `wonder_values` get Armada runs
  (`docs/card-strength.md`, `docs/wonder-strength.md`).

### 2.6 GUI, i18n, docs

* `play/static/index.html`: a shipyard panel per city, a "+ ship" toggle on build, wonder and
  sell (showing the combined cost), the island choice dialog, and naval tokens in the score
  table.
* German translations for every new string and card (`tests/test_i18n.py` enforces this).
* README expansion table, CLAUDE.md architecture notes, ROADMAP, and a CHANGELOG entry.

## 3. Milestones

Each milestone is a PR that leaves `make test` green and non-Armada games bit-identical.

1. **Data transcription.** `armada.json`: the 7 shipyards (costs + wonder track), the shared
   track spaces, 24 cards, 27 islands and Siracusa A/B, from the physical components or the
   effect sheets. Effects that do not exist yet go in as specs with the new type names, behind
   `implemented=False`. Tests: content loads, the counts match (24 / 27 / 7), and deck sizes
   split into 8 / 9-card hands for 3–7 players (`check_decks`).
2. **Shipyards and Naval Construction.** Player fleet state, `Action.naval`, combined payment,
   status hooks (naval strength, commercial level, blue VP), selling for yellow, record and
   replay. Tests: payment memo key, the chained and free cases, the discard-build ban, and a
   round trip through the record.
3. **Taxes, pirates and coin losses.** `_lose_coins`, highest tax per turn, commercial-level
   protection, debt with Cities. Tests follow the rulebook's tax example.
4. **Naval conflicts and boarding.** `_resolve_naval` with every tie case from the rulebook,
   boarding conflicts with diplomacy ignored, the purple cap, and the new score columns.
5. **Islands and fleet choices.** `EXPLORE` (solo draw of 4, shared split, return and
   reshuffle), island-draw cards, `ADVANCE_FLEET`, Siracusa, recorded offers, and
   `sample_world` resampling. Then flip `implemented=True`.
6. **Observation, action space, encoder.** `CityView` / `Observation` fields, the naval action
   block, `table-v3`, and the hidden-information tests.
7. **Agents.** Heuristic, board-aware and baseline support; bench tuning; strength and
   wonder-value reports.
8. **GUI and i18n.** Shipyard panel, naval toggle, island dialog, translations.
9. **Training.** Continue the universal model on `table-v3` with Armada in the expansion mix
   (`sweeps/`), then evaluate head to head against the bundled model on non-Armada tables to
   check for regressions.

Milestones 2–5 can be played and tested with scripted agents before any encoder work.
Milestone 1 is blocked on the component data (see below).

## Open questions

Check these against the official FAQ before implementing. Each is a test case.

1. **Component data**: compiled in [`armada-data.md`](armada-data.md). Still missing: the concrete costs and
   Wonder tracks of 6 of the 7 shipyards. The pattern is known, and one board is complete. A few card
   details are flagged there (Emporium targets, Pier's boarding target).
2. **Pirates** (resolved by the effects sheet): the victim loses coins *equal to* their commercial level. Taxes
   lose `tax − level`. Foggy Island makes its owner immune to both. Open: does commercial level also protect
   against Cities plunder? The sheet ties it only to Taxes and Pirates, so presumably not.
3. **Wonder stage + Naval Construction for stages with their own fleet moves** (Siracusa): can
   both happen in one turn on the same track?
4. **Siracusa**: its stages are built in any order (wiki; reuse the Wonder Pack's `any_order`). Can its "move a ship" stages pass a tax or exploration space
   (yes, presumably: every advance triggers the space)?
5. **Simultaneous exploration with different levels** and a level deck that runs low (fewer
   than 4 cards for a solo explorer).
6. **Babylon B's 7th/8th card** and **Olympia's free build** with a naval upgrade: both are
   builds, so presumably allowed (the rulebook names Olympia explicitly).
7. **Leaders interactions**: does a recruited leader count as a "card" for Naval Construction?
   No: white is not one of the four colours. Do the Leaders' coin powers (e.g. Nero on boarding
   victories) apply?

## Later

* The 4/6-player **team variant** (shared information, double military tokens, 1-token
  boardings, team scores) needs team-aware observations and rewards. It is out of scope for the
  first pass.
* Armada-aware search (`SearchAgent`) and analysis labels for naval moves.
