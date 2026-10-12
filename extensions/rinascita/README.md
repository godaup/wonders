# Rinascita

*From plague to enlightenment.* A data expansion in three movements, one per Age:

| Age | Theme | What changes |
|---|---|---|
| I | **The Black Death** | dear cards with drawbacks: debt you take on yourself, coins everybody else loses, quarantine that keeps you out of a war |
| II | **Patronage** | the new **Arts** colour (rose, scored with civilian points), income for building Arts, generosity towards neighbours, treasury turned into fame |
| III | **Enlightenment** | flexible science (a wildcard and a copy, both at a price), science-set bonuses, and an Inquisition that bleeds your neighbours |

Every Age adds one card per player (seven different cards: three in every game, one more for each
player from four to seven), so hands have 8 cards (9 with Cities). Two guilds join the Age III
guild draw, and three wonder boards (both sides) join the board pool:

* **The Duomo of Florence** — cheaper Arts (A) or a patron's board: generosity, income and fame from the Arts (B);
* **Uraniborg**, Tycho Brahe's observatory — science-set bonuses (A) or copying a neighbour's science (B);
* **The Plague Column of Vienna** — a war you sit out and defeats turned into coins (A), or plunder and debt for your neighbours (B).

Everything is built from the game's existing effect types: no new rules. German card names are in
`de.json`.

## Balance

Tuned to sit at or a little above the official cards (a "tiny bit spicier"), measured with the
game's strength tools (`docs/card-strength.md`): every card was simulated where it naturally comes
up in games with the whole expansion (paired build-vs-discard from the same position), and every
board against the official sides in otherwise identical deals. The values below are in VP of final
score margin, cards relative to discarding them and boards relative to an average official side.

BALANCE_TABLE

## Play it

```
WONDERS_EXTENSIONS=extensions wonders play -e base -e rinascita -a heuristic,board,greedy
WONDERS_EXTENSIONS=extensions wonders gui     # then pick Rinascita in the menu
```
