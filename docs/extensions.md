# Extensions

Add your own **expansions**, **card effects**, **agents**, **trained models** and **translations**
without changing this repository.

| kind | what you ship | as a folder (data) | as a plugin (code) |
|---|---|---|---|
| expansion | cards and wonder boards (JSON) | ✓ | ✓ |
| model | a network checkpoint (`.pt`), usable as `neural:<name>` | ✓ | ✓ |
| locale | a translation: a new language, or your cards' names in one (JSON) | ✓ | ✓ |
| card effect | a new effect type for card JSON | — | ✓ |
| agent | a scripted agent for the CLI, benchmarks and the GUI | — | ✓ |

**Trust rules.** Folders hold data only, so they are loaded automatically (checkpoints load with
`weights_only=True`, they cannot run code). Code only comes from Python packages you install yourself
(`pip install …`); the game never downloads anything. Broken extensions are skipped with a warning.

**Sharing one.** Extensions can be contributed to this repository by pull request, where they are
checked automatically: data folders go to [`extensions/`](../extensions/README.md), plugin packages
to [`plugins/`](../plugins/README.md) and translations of the game to
[`wonders/i18n/`](../wonders/i18n/README.md) (overview in [CONTRIBUTING](../CONTRIBUTING.md)).
`wonders ext check` runs the same check locally.

## Quick start: the expansion builder

The easiest way to make an expansion is the **expansion builder** in the browser GUI: run `wonders gui`
and click "Expansion builder". It edits everything below in forms: card colours, cards (cost, copies
per player count, chains, effects picked from the effect catalogue with a live card preview), wonder
boards (both sides, stages, board powers, with a live board preview) and the meta rules. "Check" shows
problems on the item they belong to; "Save" writes the extension folder and loads it, so the game menu
offers it at once; "Test & share" plays it against the computer opponents, runs a balance check
(scripted agents play with and without the expansion on the same seeds: score shifts, how often each
new card is built and how its builders place, how the new wonders do) and downloads the folder as a
zip. The builder writes exactly the format below, so a builder-made expansion can be edited by hand
and vice versa, and contributed like any other.

## Quick start: a folder extension

```
wonders ext new my-expansion --name "My Expansion" --author "You"   # creates ~/.wonders/extensions/my-expansion
wonders ext validate ~/.wonders/extensions/my-expansion
wonders ext check ~/.wonders/extensions/my-expansion   # stricter: what a pull request must pass
wonders ext list
wonders play -e base -e my-expansion -a heuristic,board,greedy
```

The folder (default home `~/.wonders/extensions`, or `$WONDERS_EXTENSIONS` — several folders
separated by `os.pathsep`, e.g. `WONDERS_EXTENSIONS=extensions:$HOME/.wonders/extensions` to use a
checkout's community extensions too) contains a manifest and the files it names:

```
my-expansion/
├── manifest.json
├── cards.json          # an expansion: cards and wonders
├── mynet.pt            # optional: a trained model
└── de.json             # optional: German strings
```

`manifest.json`:

```json
{
 "id": "my-expansion",
 "name": "My Expansion",
 "version": "0.1.0",
 "author": "you",
 "license": "CC-BY-4.0",
 "description": "A custom expansion.",
 "rules_version": 2,
 "expansions": [{"id": "my-expansion", "name": "My Expansion", "file": "cards.json"}],
 "models": [{"name": "mynet", "file": "mynet.pt", "description": "Trained on my expansion"}],
 "locales": [{"lang": "de", "file": "de.json"}]
}
```

`rules_version` is the engine's rules version the extension was made for
(`wonders.ruleset.RULES_VERSION`); `wonders ext list` flags a mismatch.

### Cards and wonders

An expansion file uses the same format as the bundled `wonders/cards/data/*.json`:

```json
{
 "cards": [
  {"name": "Mint", "color": "yellow", "age": 2, "at_players": [3, 5],
   "cost": {"resources": {"ore": 1, "glass": 1}},
   "effects": [{"type": "coins", "amount": 5}],
   "chain_requires": "Marketplace"}
 ],
 "wonders": [
  {"name": "The Great Library", "side": "A", "starting_resource": "paper",
   "stages": [{"cost": {"resources": {"stone": 2}}, "effects": [{"type": "victory_points", "points": 3}]}]}
 ]
}
```

* `color`: `brown`, `grey`, `yellow`, `red`, `blue`, `green`, `purple` (set `"guild": true`), `white`
  (leaders), `black` (city cards), or a colour your expansion defines (below).
* `at_players` lists the player counts from which a copy is in the deck (one entry per copy). Guilds
  and black cards are drawn at random instead.
* Resources: `wood`, `stone`, `ore`, `bricks` (clay), `glass`, `linen` (loom), `paper` (papyrus).
* An effect's optional `description` is the text the card renderer shows.

### New card colours

An expansion can bring its own card types. Declare them under `"colors"` in the expansion file and
use the id as a card's `color`, in effect specs that take colours (`tally`, `color_sets`,
`build_free`, `coin_discount`, `build_income`, `resource_discount`) and in other expansions loaded
after yours:

```json
{
 "colors": [
  {"id": "harbour", "label": "Harbour", "fill": "#1f8a8a", "score": "civilian", "tradeable": true}
 ],
 "cards": [
  {"name": "Dock", "color": "harbour", "age": 1, "at_players": [3, 5],
   "effects": [{"type": "production", "resources": ["wood"]}, {"type": "victory_points", "points": 1}]}
 ]
}
```

| key | meaning | default |
|---|---|---|
| `id` | lowercase letters, digits, `-` and `_` | required |
| `label` | the card type shown on cards and as the GUI's group name (translate it in a locale file) | the id, capitalised |
| `fill` | the card colour, `#rrggbb` | grey |
| `score` | the scorecard row its cards' victory points go to: `civilian`, `commercial`, `guilds`, `leaders` or `cities` | `commercial` |
| `tradeable` | neighbours may buy its cards' production, as for brown and grey cards | `false` |

Cards of a new colour are dealt like other Age cards (by `at_players`); everything else a colour does
comes from its cards' effects. The official colours cannot be redefined, and two expansions that
define the same id must define it identically. The networks' encoders keep one feature per official
colour, so a new colour never changes a model's input size (the bundled network still refuses new
cards; scripted agents play them).

### Meta rules

The numbers the engine itself fixes can be changed by an expansion under `"rules"` in its file,
either set to a value or changed by an amount (`{"add": n}`, negative to lower it):

```json
{"rules": {"starting_coins": {"add": 2}, "discard_coins": 2}, "cards": [...]}
```

| rule | base game | meaning |
|---|---|---|
| `starting_coins` | 3 | coins every player starts with (Leaders sets 6) |
| `discard_coins` | 3 | coins for discarding a card |
| `coins_per_point` | 3 | at the end, one victory point per this many coins |
| `science_set_points` | 7 | victory points per complete set of the three science symbols |
| `victory_age_1` / `_2` / `_3` | 1 / 3 / 5 | victory points of a military victory token in each Age |
| `defeat_points` | -1 | victory points of a military defeat token |
| `extra_guilds` | 2 | Age III deals as many guilds as players plus this number |

Rules combine in a fixed order, whatever order the expansions were chosen in: the base game, the
bundled expansions (Leaders, Cities), then extension expansions by id. Starting from the base value,
each `{"add": n}` changes the value so far and each plain number replaces it, so changes from several
expansions add up (with Leaders, which sets 6 starting coins, `{"add": 2}` gives 8), and the result is
clamped to the rule's range. The older manifest key `starting_coins` still works as a shorthand for
setting it. The scripted agents value military tokens with the base game's numbers.

`wonders ext validate` checks that every card and wonder builds, that new colours are well-formed, that names do not clash with the
bundled cards or each other, and that chains point to existing cards. `wonders ext check` adds the
contribution rules (complete manifest, data-only files, size limits), warns when new cards change
hand sizes (hands are deck size // players, so extra Age cards leave cards undealt — guilds and
black cards are drawn and do not), and plays and replays games with the expansion.

Note: the bundled network only knows the bundled cards, so it refuses games with new cards; scripted
agents play them out of the box (they read effects through the engine's uniform hooks).

### Effect types

Every effect type a card or wonder stage can use, with an example from the bundled content:

| type | example |
|---|---|
| `production` | `{"type": "production", "resources": ["wood"]}` |
| `multi_production` | `{"type": "multi_production", "slots": [["wood"], ["stone", "ore"]]}` |
| `coins` | `{"type": "coins", "amount": 5}` |
| `military` | `{"type": "military", "shields": 1}` |
| `science` | `{"type": "science", "symbol": "compass"}` |
| `science_wild` | `{"type": "science_wild", "count": 1}` |
| `trade` | `{"type": "trade", "side": "right", "category": "raw"}` |
| `victory_points` | `{"type": "victory_points", "points": 3}` |
| `tally` | `{"type": "tally", "what": "colors", "scope": "both", "colors": ["brown"], "coins_each": 1}` |
| `special` | `{"type": "special", "power": "my_power", "description": "text only: no effect on play"}` |
| `debt` | `{"type": "debt", "amount": 1, "scope": "neighbours"}` |
| `diplomacy` | `{"type": "diplomacy", "count": 1}` |
| `plunder` | `{"type": "plunder", "amount": 1, "scope": "others"}` |
| `grant` | `{"type": "grant", "amount": 1, "scope": "others"}` |
| `gain_token` | `{"type": "gain_token", "value": 1}` |
| `play_last_card` | `{"type": "play_last_card"}` |
| `free_build` | `{"type": "free_build"}` |
| `build_from_discard` | `{"type": "build_from_discard"}` |
| `copy_guild` | `{"type": "copy_guild"}` |
| `color_sets` | `{"type": "color_sets", "colors": ["blue", "red", "green"], "vp_each": 3, "description": "3 VP per set of one blue + one red + one green card"}` |
| `science_set_bonus` | `{"type": "science_set_bonus", "points": 3, "description": "+3 VP per completed set of 3 science symbols"}` |
| `science_copy` | `{"type": "science_copy", "count": 1, "description": "at game end, copy one science symbol from a neighbour"}` |
| `build_income` | `{"type": "build_income", "coins": 2, "color": "yellow", "description": "+2 coins each time you build a yellow card"}` |
| `victory_income` | `{"type": "victory_income", "coins": 2, "description": "+2 coins each time you gain a military victory token"}` |
| `redirect_defeats` | `{"type": "redirect_defeats", "description": "your defeat tokens are given to the winning neighbour"}` |
| `resource_discount` | `{"type": "resource_discount", "target": "green", "amount": 1, "description": "green cards cost 1 fewer resource"}` |
| `build_free` | `{"type": "build_free", "colors": ["white"], "description": "recruit all your future leaders for free"}` |
| `bank_purchase` | `{"type": "bank_purchase", "count": 1, "description": "once per turn buy any 1 resource from the bank for 1 coin"}` |
| `produce_unowned` | `{"type": "produce_unowned", "description": "once/turn produce 1 resource you do not already produce"}` |
| `produce_owned` | `{"type": "produce_owned", "description": "once/turn produce 1 extra of a resource you already produce"}` |
| `starting_resource_discount` | `{"type": "starting_resource_discount", "amount": 1, "description": "buy your starting resource 1 cheaper from a neighbour"}` |
| `cash_defeats` | `{"type": "cash_defeats", "coins_each": 2, "description": "gain 2 coins per defeat token, then discard all your defeat tokens"}` |
| `plunder_per` | `{"type": "plunder_per", "what": "victory_tokens", "amount": 1, "scope": "others", "description": "other players lose 1 coin per Victory token they hold"}` |
| `coin_discount` | `{"type": "coin_discount", "colors": ["white"], "amount": 2, "neighbours_amount": 1, "description": "leaders cost 2 less here, 1 less for neighbours"}` |
| `leader_draw` | `{"type": "leader_draw", "count": 4, "description": "draw 4 leaders from the box"}` |
| `extra_recruit` | `{"type": "extra_recruit", "count": 1, "description": "recruit an extra leader now"}` |
| `copy_leader` | `{"type": "copy_leader", "count": 1, "description": "copy a leader recruited by a neighbour"}` |

New mechanics need a new effect type, which is code: see plugins below.

## Plugins: effects, agents and anything else

A plugin is an installed Python package with an entry point in the `wonders.extensions` group that
points to a `register(registry)` function. Optionally, the module defines `MANIFEST` (same fields as
`manifest.json`; the package version is used by default). [`plugins/example`](../plugins/example) is
a complete, installable one.

`pyproject.toml` of your package:

```toml
[project.entry-points."wonders.extensions"]
my-plugin = "my_plugin:register"
```

`my_plugin/__init__.py`:

```python
from wonders.agents import GreedyAgent
from wonders.agents.bench import AgentSpec
from wonders.cards.schema import EffectSchema, Param
from wonders.effects.base import Effect

MANIFEST = {'id': 'my-plugin', 'name': 'My plugin', 'rules_version': 2}


class Tithe(Effect):
    """Victory points per coin held at the end (an example mechanic)."""

    def __init__(self, per: int):
        self.per = per

    def victory_points(self, ctx):
        return ctx.player.coins // self.per


TITHE = EffectSchema(  # optional: lets the GUI's expansion builder offer a form for the effect
    'Tithe', 'Points', '1 victory point per this many coins at the end.', (Param('per', 'int', 'Coins', required=True),)
)


def register(registry):
    registry.register_effect('tithe', lambda spec: Tithe(spec['per']), schema=TITHE)
    registry.register_agent(AgentSpec.of('greedy-plus', GreedyAgent), blurb='My greedy variant.')
    # registry.register_expansion('my-cards', 'My cards', '/path/to/cards.json')
    # registry.register_model('mynet', '/path/to/mynet.pt', 'description')
    # registry.register_locale('de', {'key': 'Text'})
```

An effect type registered without a `schema` (a `wonders.cards.schema.EffectSchema`: label, group,
help and its parameters) still works everywhere; the expansion builder then edits its fields as JSON.

Effects compose behaviour through the uniform hooks of `wonders.effects.base.Effect`
(`production`, `shields`, `science`, `victory_points`, `coins_on_build`, `on_build`, …); the engine
never special-cases effect classes, so a new effect works everywhere — scoring, payments, the card
renderer (via its `description`), the observation and the scripted agents.

## Translations

A locale is a flat JSON object that maps the **English text** to its translation, exactly like the
bundled `wonders/i18n/de.json` (the reference for every key the game uses):

```json
{
  "_language": "Français",
  "_decimal": ",",
  "Start game": "Commencer la partie",
  "{n} Shields": "{n} boucliers",
  "Loom|resource": "Tissu",
  "Workshop": "Atelier"
}
```

* `_language` is the name in the language picker; `_decimal` the decimal separator; `_ordinal` (e.g.
  `"{n}."`) how ranks are written.
* Keep every `{field}` of the English text (a test checks the bundled catalogs for this).
* `"text|context"` keys separate words with two meanings (`Loom` the card, `Loom|resource` the
  resource); a lookup falls back to the bare text, and anything missing stays English.
* A locale for an existing language extends it — ship the names of your expansion's cards in German
  next to the expansion itself (`"locales": [{"lang": "de", "file": "de.json"}]`).

The GUI offers every language found (picker in the header, `?lang=de` in the URL, the browser's
language by default); `wonders gallery --lang de` renders the cards in it. Game records, agents and
the API keep the English names, so a game played in one language replays in any other.

## Game records

A record of a game with extension content stores each extension's id, version and content hash
(`GameRecord.extensions`). Replaying it with a different or missing extension warns that the replay
may differ; `GameRecord.check_extensions()` lists the differences.

## Reference

* `wonders ext list | validate FOLDER | new ID [--name] [--author] [--dir]`
* `wonders ext check TARGET…` — the contribution check for extension folders, installed plugin
  packages and locale files; `wonders ext scorecard FOLDER|PT [--model] [--cells] [--games]` — a
  model against the bundled network; `wonders ext changed | summary | leaderboard` — used by the
  contribution workflows.
* `wonders.extensions`: `REGISTRY`, `ensure_loaded()`, `load_folder()`, `load_plugin()`, `validate()`,
  `scaffold()`; `wonders.contrib`: `check()`, `scorecard()`, `changed()`, `update_leaderboard()`
* `WONDERS_EXTENSIONS` — the extensions home (default `$WONDERS_HOME/extensions`, i.e.
  `~/.wonders/extensions`; several folders separated by `os.pathsep`); `WONDERS_NO_EXTENSIONS=1`
  disables discovery.
