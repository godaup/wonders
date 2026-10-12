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

| card | Age | colour | value | official range (same Age and colour) |
|---|---|---|---:|---|
| Quarantine Station | I | blue | +0.9 ± 0.4 | +0.9 to +2.0 |
| Plague Doctor | I | green | +1.8 ± 0.5 | +2.0 to +2.2 |
| Flagellants | I | red | +3.1 ± 0.4 | +2.3 to +2.6 |
| Grave Robbers | I | yellow | +2.0 ± 0.4 | +1.1 to +2.0 |
| Abandoned Farmstead | I | brown | +3.0 ± 0.5 | +2.4 to +5.8 |
| Plague Pit | I | blue | +2.0 ± 0.6 | +0.9 to +2.0 |
| Rat Catchers | I | red | +1.5 ± 0.8 | +2.3 to +2.6 |
| Fresco Workshop | II | arts | +2.5 ± 0.2 | -1.2 to +5.1 |
| Sculptor's Studio | II | arts | +2.4 ± 0.2 | -1.2 to +5.1 |
| Patron's Palazzo | II | arts | +2.7 ± 0.3 | -1.2 to +5.1 |
| Medici Bank | II | yellow | +2.1 ± 0.3 | +1.1 to +4.0 |
| Public Festival | II | blue | +1.8 ± 0.3 | +1.6 to +2.8 |
| Theatre of the Republic | II | arts | +2.7 ± 0.3 | -1.2 to +5.1 |
| Condottieri | II | red | +2.8 ± 0.7 | +3.4 to +4.0 |
| Orrery | III | green | +0.8 ± 0.1 | +0.9 to +1.4 |
| Encyclopédie | III | green | +1.5 ± 0.1 | +0.9 to +1.4 |
| Royal Academy | III | blue | +4.1 ± 0.1 | +3.1 to +4.8 |
| Salon | III | arts | +3.1 ± 0.1 | -2.9 to +5.5 |
| Printing Press | III | yellow | +1.7 ± 0.1 | +1.7 to +4.1 |
| Inquisition | III | red | +2.7 ± 0.3 | +2.6 to +3.0 |
| Grand Tour | III | yellow | +1.6 ± 0.2 | +1.7 to +4.1 |
| Patrons' Guild | III | purple | +2.4 ± 0.2 | +1.6 to +5.5 |
| Academicians' Guild | III | purple | +3.9 ± 0.1 | +1.6 to +5.5 |

| board side | value | official sides |
|---|---:|---|
| The Duomo of Florence A | +1.0 ± 0.6 | -5.0 to +5.4 |
| The Duomo of Florence B | -1.2 ± 0.7 | -5.0 to +5.4 |
| Uraniborg A | +0.7 ± 0.7 | -5.0 to +5.4 |
| Uraniborg B | -0.0 ± 0.7 | -5.0 to +5.4 |
| The Plague Column of Vienna A | -1.4 ± 0.6 | -5.0 to +5.4 |
| The Plague Column of Vienna B | +3.0 ± 0.7 | -5.0 to +5.4 |

From 900 watched games (cards) and 450 paired deals per side (boards), 3 to 7 players, `board` agents
in every seat. Age I is deliberately harsh: with the same boards, players end Age I with as many coins as
without the expansion despite an extra play (plunder drains everyone) and 0.24 debt each.

## Play it

```
WONDERS_EXTENSIONS=extensions wonders play -e base -e rinascita -a heuristic,board,greedy
WONDERS_EXTENSIONS=extensions wonders gui     # then pick Rinascita in the menu
```
