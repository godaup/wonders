# Game editions

7 Wonders exists in two editions: the first (2010) and the second (2020). The second keeps the rules but revises
cards, chains and every wonder board, and its expansions were reprinted with changes of their own. This repo
implements the first edition throughout and is moving to the second. Both stay playable: saved games, records and
trained models refer to first-edition content by name.

## How an edition is chosen

Content is an expansion id per edition. `base` is the first-edition base game and `base2` the second (data file
`wonders/cards/data/base2.json`). Each `Expansion` in `wonders/ruleset.py` has an `edition` (`None`: any edition,
e.g. extension expansions) and a `family` (`base2` is an edition of `base`).

* `Ruleset.of(*ids)` takes exactly one base game. Without one it implies `base`, so every existing caller,
  record and saved game means the first edition. It refuses to mix editions (`base2` with the first edition's
  Leaders) or two base games.
* `Ruleset.edition` is the base game's edition. `Ruleset.uses('cities')` and `has_family(ids, 'cities')` answer
  for every edition, so the engine and encoders never test ids like `'cities' in expansions`.
* `with_edition(ids, edition)` maps the ids of any edition to one edition (`['leaders'], 2` raises "Leaders is
  not available in the second edition"). The GUI (`GameConfig.edition`, the menu's edition select) and the CLI
  (`wonders play|bench --edition 2`) use it.
* Records, observations and checkpoints carry the expansion tuple, so they carry the edition: a second-edition
  record says `base2` and replays with that content.
* The two editions share card names with different values (Altar is worth 2 or 3 points). The GUI's image routes
  take `&edition=` and render the card of that edition. A network trained on one edition refuses the other: its
  card and wonder vocabulary does not cover the other edition's (Well, Ludus, Castrum, the second-edition
  boards).

## The second-edition base game (`base2.json`)

70 cards and 10 guilds (148 Age cards: 49 / 49 / 50 at 7 players), 7 boards with a day side (`A`) and a night
side (`B`). The board names are the ones printed on the second edition's boards (Alexandria, Babylon, Ephesos,
Gizah, Halikarnassos, Olympia, Rhodos), so they never clash with the first edition's.

Sources:

| What | Source |
|---|---|
| card names, effects, minimum player counts, chains | official "List of cards and chains" sheet, `sev-en02-chain-list` (Asmodee CDN) |
| rules, deck composition (148 cards), guilds = players + 2 | official rulebook, `sev-en02-rules` (Asmodee CDN) |
| wonder boards (costs, effects, starting resources) | the 14 board images (7 Wonders Fandom wiki), matching the wiki's board data |
| card costs | 7 Wonders Fandom wiki tables (`List of Age Cards/Overview`, `List of Cards`) |
| cross-checks | Assorted Meeples' 1st-vs-2nd-edition change list; the `v2_cards.json` of paulxiep/portfolio and the TabletopGames framework (both mix in first-edition costs) |

Every card's player counts on the wiki match the official sheet. **Four costs are unverified**: the wiki and the
two engines disagree, and the official sheet shows no costs. `base2.json` uses the wiki's values:

| Card | wiki (used) | other sources (= first edition) |
|---|---|---|
| Town Hall | 3 stone, glass | 2 stone, ore, glass |
| Arena | 2 clay, ore | 2 stone, ore |
| Circus | 3 clay, ore | 3 stone, ore |
| Craftsmens Guild | 2 stone, ore | 2 stone, 2 ore |

A look at the four printed cards settles them.

### What changed from the first edition

* **Cards**: Well replaces Pawnshop; Altar and Theater give 3 points, Temple 4. New are Ludus (Age III yellow:
  3 coins and 1 point per red card, 5+/7+) and Castrum (Age III red, 3 shields, 4+/7+), and the Decorators
  Guild (7 points if your wonder is complete) replaces the Strategists Guild. Town Hall, Arena, Circus and
  Arsenal lose a copy; Fortifications costs 3 ore and clay, the Spies Guild 2 clay and glass, the Builders
  Guild one stone more.
* **Chains**: Altar → Pantheon and Theater → Gardens skip Age II, Well → Statue, Temple chains nowhere.
* **Boards**: new costs everywhere; Babylon starts with wood and Olympia with clay; Babylon's night side has two
  stages. New powers on Olympia: the first card of each colour is free (day), and the first and the last card
  of each Age are free (night, stages 1 and 2). Olympia no longer copies a guild, and Halicarnassus' night side
  builds from the discard on all three stages.

New effects: `free_first_of_color` (read by `Player.builds_free`), `free_turn_build` (`turn: first|last`;
`Game._free_turn` offers a `FREE` action on that turn of each Age without spending the per-Age allowance) and
`wonder_complete`.

## Second-edition Leaders (`leaders2.json`)

Sources: the official rulebook and effect sheet (`7le-en02-rules`, `7le-en02-effects-description`), the wiki's
`Leader Cards` table (55 leaders, matching the rulebook's count) and the board images.

* **Leaders**: 55, in three decks that the effect sheet lists power by power. 34 are *standard*, 15 *expert*, and
  6 are *bonus* leaders that join only together with Cities (`"requires": ["cities"]`, which `load_content` checks
  by family). The 36 leaders the first-edition file has keep their costs and powers; their specs are reused.
* **New mechanics**:
  * Nine leaders cost the Age: 1, 2 or 3 coins (`"age_cost": true`, `Card.coin_cost(age)`). Abu Simbel scores
    6 for such a leader.
  * New effects:
    * `science_replace` (Aganice), `science_most` (Enheduanna): handled by `science_score`.
    * `majority` (Phryne, Cornelia, Euryptyle, Theano, Makeda), `no_defeats` (Cynisca), `token_pairs` (Gorgo),
      `only_leader` (Agrippina).
    * `token_purge` (Telesilla): the others drop their lowest victory token.
    * `reserve_bonus` (Berenice): coins from the reserve go through `Player.gain`, and the bonus is paid once
      at the end of each turn.
    * `stage_plunder` (Octavia, together with `build_income`'s new `wonder` flag).
  * New flags on existing effects: `gain_token`'s `current_age` (Nitocris), `plunder`'s `per_age` (Arsinoe) and
    `free_build`'s `colors` (Caligula: one black card per Age).
* **Setup rules** (meta rules): the draft passes right (`leader_draft_left` 0). At Age III recruitment the
  unchosen leader goes back to the box (`return_last_leaders` 1), except for a player on Roma's night side.
* **No guilds**: the first edition's four Leaders guilds are gone.
* **Boards**: Roma and Abu Simbel. Roma's night stage 2 costs glass (linen in the first edition). Abu Simbel's
  night stages give no points; the wiki template carries hidden stale values, but the board image shows none.

## Second-edition Cities (`cities2.json`)

Sources: the official rulebook and effect sheet (`7ci-en02-rules`, `7ci-en02-effects-description`), the wiki's card
list and the board images.

* **Cards**: the 42 black cards (14 per Age). Their names, Ages, costs and effects are identical to `cities.json`
  (checked card by card), which was evidently compiled from second-edition sources.
* **Boards**: new Byzantium and Petra. Petra's coin stages cost 5 and 10 (7 and 14 in the first edition), and
  Byzantium's stages are new.
* **Rules**: unchanged (8-card hands, N random black cards per Age, debt for coins not paid, diplomacy).
* **Known approximations**:
  * Taking debt by choice although you could pay is not offered (it is never better).
  * Telesilla's victims always give up their lowest victory token.

## Next

Armada's second edition (`armada2`; see `docs/armada.md`). Then agents and value tables tuned on the second
edition, a model trained on it, and switching the default edition.
