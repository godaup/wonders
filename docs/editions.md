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

## Next

Leaders, Cities and Armada exist as second-edition reprints with changes of their own (`leaders2`, `cities2`,
`armada2`). The official second-edition rulebooks are on the Asmodee CDN (`7le-en02-rules`, `7ci-en02-rules`,
`7ar-en02-rules`). After that come agents and value tables tuned on the second edition, a model trained on it,
and switching the default edition.
