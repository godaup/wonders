# Armada: compiled component data

This is milestone 1 of [`docs/armada.md`](armada.md): the cards, islands, shipyards and wonder, compiled from
web sources. A machine-readable draft in the repo's card-JSON format is
[`docs/armada/armada.draft.json`](armada/armada.draft.json). The package does not load it: effect types such as
`naval_shields` and `advance_fleet` do not exist yet. Moving it to `wonders/cards/data/armada.json` is part of
milestone 2. Resources use the repo's names: cloth = `linen`, papyrus = `paper`, clay = `bricks`.

## Edition

The repo implements the **first edition** of 7 Wonders: `base.json` has Pawnshop and none of the second edition's
new cards, and Leaders and Cities were checked against first-edition rulebooks. Armada was published for both
editions, so the data below is **first edition (2018)**. Every second-edition (2020) difference is listed next to
it (`name_en02` and `note` in the draft).

| | 1st edition (2018, "EN01") | 2nd edition (2020, "EN02") |
|---|---|---|
| Terms | Dockyard, Incursion, Ground Conflict | Shipyard, Boarding, Military Conflict |
| Boards / players | 8 Dockyards, 32 ships; 3–7 players + an 8-player variant (7 cards dealt, mainly for teams) | 7 Shipyards, 28 figures; 3–7 players |
| Naval tokens | 48: victory 11×1, 12×3, 7×5, 1×7; defeat 7×−1, 5×−2, 5×−3 | 40: 22 victory, 18 defeat |
| Naval conflict | last −1/−2/−3, first 3/5/7, second 1/3/5, **third +3 in Age III only**; a player with a defeat token cannot also win one; ties shift down a rank (see below) | last / first / second only; tied for first take the second-place token |
| Yellow coins, spaces 2–5 | **3 / 4 / 5 / 6** in the board figure (the rule text's examples say 2 and 4) | 2 / 3 / 4 / 5 |
| Lost Island | Naval Constructions via the Wonder are **free and of any colour** | no wonder fleet; each wonder stage advances a fleet of your choice for free |
| Wondrous Harbor | +1 coin per Naval Construction (free or paid) | +1 coin after each fleet advance |
| Turn order | … 3. effects except coin losses, **4. island exploration**, 5. discard builds, 6. coin losses | … 3. card effects, 4. shipyard effects, 5. discard builds, 6. coin losses |
| Incursion / Boarding FAQ | one extra conflict at most per pair of players per Age (Dock + Jetty against the same player = 1) | – |
| Card names, costs, effects | identical, except names (table below) and Jetty's target | |

**Naval ties (1st edition)**: everyone tied for weakest takes the defeat token. Players tied for strongest take the
next rank's reward: the second-place token, and in Age III the second-strongest then gets the third-place +3. Players
tied for second get nothing in Ages I–II. In Age III, they get +3 only if exactly one player is strongest. Ties for
third get nothing. If every player is tied, nobody gets a token.

## Sources

| Id | Source | Edition | Used for |
|---|---|---|---|
| **R1** | Armada rulebook, 2018 (`51-7-wonders-armada-rulebook.pdf`, 1jour-1jeu) | 1st | rules, token counts, every card image (costs), island effects, one complete board (p. 3) |
| **S1** | Siracusa rulesheet, 2018 (`06-7-wonders-armada-siracusa-rulebook.pdf`, 1jour-1jeu) | 1st | Siracusa effects, any-order rule (no costs printed) |
| **R2** | Armada rulebook `7ar-en02-rules` (Asmodee CDN) | 2nd | rules; the same board as **R1** |
| **E2** | "Description of New Effects" `7ar-en02-effects-description` (Asmodee CDN) | 2nd | symbol meanings |
| **F** | 7 Wonders Fandom wiki (cards, islands, Shipyard Board, Siracusa; via the MediaWiki API) | 2nd | island list, cost pattern, Siracusa stage costs |
| **S2** | Siracusa Day / Night board images (wiki) | 2nd | Siracusa stage costs |
| **U** | UltraBoardGames rules summary | 1st | cross-check of names and rules |

`karlaugustnybo/7-wonders-ai` (Rust, MIT) was **not used**, and its Armada data is not an edition variant. Its costs
match neither edition's card images. Its blue card "Admiralty" does not exist: UBG prints "Dry Dock Hangar" on one
line, which hides that **R1** lists six blue cards (Shipyard, Construction Yard, Dry Dock, Hangar, Captaincy, Naval
Ministry). Its Siracusa A (3 VP / advance / 7 VP) contradicts **S1**.

## Age cards (24: 8 per Age, N random per Age added to the deck)

`4+` = removed at 3 players (the five named in **R1** setup). Costs were read from the card images in **R1**
pp. 12–13 and match **F** card for card. None of the cards shows a chain.

| Age | Colour | Name (2nd ed.) | Cost | Effect | 4+ |
|---|---|---|---|---|---|
| I | blue | Shipyard (Dry Hold) | free | advance a fleet of your choice 1 space, free | |
| I | blue | Construction Yard (Construction Zone) | free | advance a fleet of your choice, free | |
| I | yellow | Pirate Cache (Pirate Hideout) | free | +3 coins; all others lose 1 coin per commercial level | |
| I | yellow | Eastern Emporium | linen | once per turn buy 1 resource for 1 coin from the player right of your right neighbour | ✓ |
| I | red | Coastal Defenses | glass | 1 naval shield | |
| I | red | Dock (Pontoon) | paper | 1 shield; Incursion ▷▷ (the player right of your right neighbour) | ✓ |
| I | green | Steersmen's Academy (Helmsman's Rest) | stone | put the top level-1 Island in play | |
| I | green | Explorer's Office (Cabinet of Explorers) | bricks | +1 science symbol: the one you have most of | |
| II | blue | Dry Dock (Ship Repair) | ore, glass | advance a fleet of your choice; 2 VP | |
| II | blue | Hangar (Boathouse) | 2 wood, linen | advance a fleet of your choice; 3 VP | |
| II | yellow | Pirate Lair (Pirate's Den) | free | +6 coins; pirates | |
| II | yellow | Western Emporium | ore, paper | **twice** per turn buy 1 resource for 1 coin from the player left of your left neighbour | ✓ |
| II | red | Fortified Port | bricks, paper | 2 naval shields | |
| II | red | Quay (Ballista) | bricks, glass, linen | 2 shields; Incursion ◁◁ (left of your left neighbour) | ✓ |
| II | green | Map Room (Cartographer's Office) | wood, bricks | put the top level-2 Island in play | |
| II | green | Society of Navigators (Navigation Firm) | stone, ore, glass | +1 science symbol (most) | |
| III | blue | Captaincy (Harbor Master's Office) | wood, stone, ore, linen | advance a fleet of your choice; 4 VP | |
| III | blue | Naval Ministry (Naval Headquarters) | 2 stone, wood, bricks, paper | advance a fleet of your choice; 5 VP | |
| III | yellow | Pirate Brotherhood (Pirate Crew) | wood, glass | 4 VP; pirates | |
| III | yellow | Customs Office (Port Customs) | 2 ore, stone, bricks | end: 2 VP per commercial level | |
| III | red | Coastal Fortifications | 2 stone, bricks, glass | 3 naval shields | |
| III | red | Jetty (Pier) | 3 wood, paper, linen | 3 shields; Incursion ▷▷ (right of your right neighbour) | ✓ |
| III | green | Naval Archives (Nautical Archives) | 2 ore, wood, paper | put the top level-3 Island in play | |
| III | green | Captain's Circle (Circle of Captains) | 2 bricks, stone, ore, linen | +1 science symbol (most) | |

Settled by **R1**:

* **Emporium targets**: the player two seats away. The wiki's "right neighbour" for both is wrong.
* **Jetty's target**: ▷▷, per its card image. The wiki's left target for Pier may be a 2nd-edition change or a wiki error.
* **Blue cards**: they move a fleet *of your choice*. With a Blue Naval Construction on the same build, two moves in
  one turn are allowed.

**Ages**: **R1** gives eight cards per Age as 2 per colour but does not name the Ages. Together with **F** (whose
card list puts Captaincy and Naval Ministry in Age II but whose board page puts them in Age III) and the VP
progression 0/0/2/3/4/5, Age III is the consistent reading.

## Islands (27: 9 per level)

The names show the level: Island = 1, Harbor = 2, Archipelago = 3.

| Lvl | Name (2nd ed.) | Effect |
|---|---|---|
| 1 | Wild Island (Overgrown Island) | produces wood **or** ore (neighbours cannot buy it) |
| 1 | Virgin Island | produces stone or bricks |
| 1 | Bronze Island | 1 shield |
| 1 | Windblown Island (Whistling Island) | 1 naval shield |
| 1 | Foggy Island | no coin losses from Taxes or Pirates |
| 1 | Topaz Island | advance the **blue** fleet, free |
| 1 | Amber Island | advance the **yellow** fleet, free |
| 1 | Lost Island | Naval Constructions via the Wonder are free and of any colour *(2nd ed. differs, above)* |
| 1 | Inhabited Island | 3 VP |
| 2 | Abandoned Harbor (Abandoned Oasis) | produces glass, paper or linen |
| 2 | Iron Harbor (Iron Oasis) | 2 shields |
| 2 | Windy Harbor (Windy Oasis) | 2 naval shields |
| 2 | Forgotten Harbor (Forgotten Oasis) | may sit out each naval conflict (its strength is ignored) |
| 2 | Ancient Harbor (Ancient Oasis) | each Naval Construction costs 1 resource less (coins in full) |
| 2 | Frothy Harbor (Stormy Oasis) | advance two different fleets, free |
| 2 | Wondrous Harbor (Wonderous Oasis) | +1 coin per Naval Construction *(2nd ed. differs)* |
| 2 | Statues Harbor (Oasis of Statues) | 5 VP |
| 2 | Pirate Harbor (Pirate Oasis) | pirates; 3 VP |
| 3 | Scarlet Archipelago | 3 shields |
| 3 | Trade Winds Archipelago (Breezy Archipelago) | 3 naval shields |
| 3 | Stormy Archipelago (Torrential Archipelago) | advance three different fleets, free |
| 3 | Golden Archipelago | end: a science symbol of your choice |
| 3 | Emerald Archipelago | end: +1 science symbol (most) |
| 3 | Sovereign Archipelago (Royal Archipelago) | 7 VP |
| 3 | Luxurious Archipelago (Lush Archipelago) | end: 2 VP per commercial level |
| 3 | Rainbow Archipelago | end: choose a colour, 1 VP per card of it in your city |
| 3 | Sheltered Archipelago | end: 2 VP per Island you own (including this one) |

## Dockyards

**Track effects** (the same on every board). Values marked "current" count while the ship is on that space; the
others fire once, on arrival.

| Space | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Red: naval shields (current) | 1 | 2 | 3 | 4 | 5 | 8 |
| Yellow: commercial level (current) | 1 | 1 | 2 | 2 | 3 | 4 |
| Yellow: on arrival, 1st-edition figure | – | +3 coins, tax 1 | +4 coins | +5 coins, tax 2 | +6 coins | tax 4; 2 naval shields (current) |
| Yellow: on arrival, 2nd edition | – | +2, tax 1 | +3 | +4, tax 2 | +5 | tax 4; 2 naval shields |
| Blue: VP (current) | 1 | 2 | 4 | 6 | 8 | 10 + 1 naval shield |
| Green: on arrival | – | explore L1 | – | explore L2 | – | explore L3 |

⚠ **The yellow coins disagree inside R1**: its board figure prints 3/4/5/6, but its rule examples say "reaches the
second upgrade space … immediately gains 2 Coins" and "upgraded four times … gains 4 Coins", which is 2/3/4/5, the
second-edition values. The physical first-edition board decides this. Until then the draft follows the figure, with
the alternative in `meta.yellow_coins_en02`.

**Cost pattern** (cost of moving *to* each space; it holds on every board):

| To space | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| red / yellow / blue | 1 brown | 1 grey | 2 brown | 1 brown + 1 grey | 2 brown | 2 grey |
| green | 1 brown | 2 coins | 2 brown | 3 coins | 2 brown | 4 coins |

**The one board known in full**: it appears in both rulebooks with identical costs (**R1** p. 3, **R2** p. 2), and
its Wonder symbol is on **red**:

| To space | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| red | wood | paper | stone, bricks | stone, glass | wood, ore | glass, linen |
| yellow | stone | linen | wood, ore | bricks, paper | stone, bricks | glass, paper |
| blue | bricks | glass | stone, ore | ore, linen | wood, bricks | paper, linen |
| green | ore | 2 coins | wood, bricks | 3 coins | stone, ore | 4 coins |

**Missing**: the other 7 first-edition boards' concrete resources and Wonder tracks. No source has them. Two
stand-ins are possible until they are transcribed: use the known board for every seat, or generate
pattern-conforming variants labelled unofficial.

## Siracusa (Armada only; starts with glass; stages in any order)

Effects are from **S1** (1st edition) and are identical to the 2nd edition. A Naval Construction of the
wonder-linked colour can be combined with a stage's free advance. **S1** prints no costs, so the costs below come
from the 2nd-edition board (**S2**, **F**) and are unverified for the first edition.

| Side | Stage | Cost (2nd ed.) | Effect |
|---|---|---|---|
| A | 1 | 2 stone, ore | advance a fleet of your choice, free; 1 naval shield |
| A | 2 | 2 ore, linen | advance a fleet of your choice, free; 3 VP |
| A | 3 | 2 wood, paper | advance a fleet of your choice, free; 1 shield |
| B | 1 | ore, bricks | +6 coins; all others lose 1 coin per commercial level |
| B | 2 | 2 ore, wood | advance two different fleets, free |
| B | 3 | 2 wood, paper | 1 shield; 1 naval shield |

## Still open

1. The yellow coin values on the first-edition board (3/4/5/6 in the figure, 2/3/4/5 in the text).
2. 7 of the 8 first-edition Dockyards (concrete costs and Wonder tracks).
3. Siracusa's first-edition stage costs.
4. The Age of each Armada card: printed on the card backs, not in **R1**. The draft follows the 2nd-edition wiki and
   the VP progression.
