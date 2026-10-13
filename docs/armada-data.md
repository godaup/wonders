# Armada: compiled component data

This is milestone 1 of [`docs/armada.md`](armada.md): the cards, islands, shipyards and wonder, compiled from
web sources. A machine-readable draft in the repo's card-JSON format is
[`docs/armada/armada.draft.json`](armada/armada.draft.json). The package does not load it: effect types such as
`naval_shields` and `advance_fleet` do not exist yet. Moving it to `wonders/cards/data/armada.json` is part of
milestone 2. Resources use the repo's names: cloth = `linen`, papyrus = `paper`, clay = `bricks`.

## Sources

| Id | Source | Used for |
|---|---|---|
| **R** | Official rulebook, `7ar-en02-rules` (Repos Production / Asmodee CDN) | rules, token values, one complete shipyard board (p. 2 figure) |
| **E** | Official "Description of New Effects" sheet, `7ar-en02-effects-description` (Asmodee CDN) | every new symbol's exact meaning |
| **F** | 7 Wonders Fandom wiki: *List of Age Cards/Overview*, *Island Cards*, *Shipyard Board*, *Siracusa* (read via the MediaWiki API) | card and island names, costs and effects, the shipyard cost pattern, Siracusa |
| **S** | Siracusa Day / Night board images (wiki file pages) | confirms Siracusa's costs and effects |
| **U** | UltraBoardGames "How to play 7 Wonders Armada" (first English printing, EN01 terms: *Dockyard*, *Incursion*) | an independent naming of all 24 cards: confirms the set, the groups and the VP values |

Names follow the current printing (EN02, matching **R** / **E**). The EN01 names from **U** are kept as `name_en01`
in the draft. The repo's convention is to paraphrase card names (see CLAUDE.md), so pick final names when the data
moves into the package.

The GitHub project `karlaugustnybo/7-wonders-ai` (Rust, MIT) also has Armada data. It was **not used**: its Siracusa A
(3 VP / advance / 7 VP) contradicts **F** and **S**, and its card costs disagree with **F**.

## Age cards (24: 8 per Age, N random per Age added to the deck)

`4+` = removed at 3 players (5 cards, matching **R**). None of the sources shows chains.

| Age | Colour | Name (EN01) | Cost | Effect | 4+ |
|---|---|---|---|---|---|
| I | blue | Dry Hold (Shipyard) | free | advance a fleet of your choice 1 space, free | |
| I | blue | Construction Zone (Construction Yard) | free | advance a fleet of your choice 1 space, free | |
| I | yellow | Pirate Hideout (Pirate Cache) | free | +3 coins; all others lose coins = their commercial level | |
| I | yellow | Eastern Emporium | linen | once per turn buy 1 resource for 1 coin from the designated player ⚠ | ✓ |
| I | red | Coastal Defenses | glass | 1 naval strength | |
| I | red | Pontoon (Dock) | paper | 1 shield; Boarding → the player right of your right neighbour | ✓ |
| I | green | Helmsman's Rest (Steersmen's Academy) | stone | draw the top level-1 Island | |
| I | green | Cabinet of Explorers (Explorer's Office) | bricks | end: +1 science symbol, the one you have most of | |
| II | blue | Ship Repair (Drydock Hangar) | ore, glass | advance a fleet of your choice; 2 VP | |
| II | blue | Boathouse (Captaincy) | 2 wood, linen | advance a fleet of your choice; 3 VP | |
| II | yellow | Pirate's Den (Pirate Lair) | free | +6 coins; pirates | |
| II | yellow | Western Emporium | ore, paper | **twice** per turn buy 1 resource for 1 coin from the designated player ⚠ | ✓ |
| II | red | Fortified Port | bricks, paper | 2 naval strength | |
| II | red | Ballista (Quay) | bricks, glass, linen | 2 shields; Boarding → the player left of your left neighbour | ✓ |
| II | green | Cartographer's Office (Maproom) | wood, bricks | draw the top level-2 Island | |
| II | green | Navigation Firm (Society of Navigators) | stone, ore, glass | end: +1 science symbol (most) | |
| III | blue | Harbor Master's Office (Naval Ministry) | wood, stone, ore, linen | advance a fleet of your choice; 4 VP ⚠ | |
| III | blue | Naval Headquarters (Admiralty) | 2 stone, wood, bricks, paper | advance a fleet of your choice; 5 VP ⚠ | |
| III | yellow | Pirate Crew (Pirate Brotherhood) | wood, glass | 4 VP; pirates | |
| III | yellow | Port Customs (Customs Office) | 2 ore, stone, bricks | end: 2 VP per commercial level | |
| III | red | Coastal Fortifications | 2 stone, bricks, glass | 3 naval strength | |
| III | red | Pier (Jetty) | 3 wood, paper, linen | 3 shields; Boarding → left of your left neighbour ⚠ | ✓ |
| III | green | Nautical Archives (Naval Archives) | 2 ore, wood, paper | draw the top level-3 Island | |
| III | green | Circle of Captains (Captain's Circle) | 2 bricks, stone, ore, linen | end: +1 science symbol (most) | |

Exact effect wording from **E**:

* **Pirates**: "all players except you lose as many Coins as their respective commercial level". So a higher
  commercial level *raises* the pirate loss. This answers open question 2 of the plan.
* **Taxes**: the loss is `tax − commercial level`.
* **Boarding**: when the card enters play, "take a Boarding token (from the reserve) and give it to the designated
  player".
* **Emporiums**: "once per turn / twice per turn, you can buy 1 resource from the designated player for 1 Coin".

## Islands (27: 9 per level)

| Lvl | Name | Effect |
|---|---|---|
| 1 | Overgrown Island | produces wood **or** ore (neighbours cannot buy it) |
| 1 | Virgin Island | produces stone or bricks |
| 1 | Bronze Island | 1 shield |
| 1 | Whistling Island | 1 naval strength |
| 1 | Foggy Island | immune to coin losses from Taxes and Pirates |
| 1 | Topaz Island | advance the **blue** fleet 1 space, free |
| 1 | Amber Island | advance the **yellow** fleet 1 space, free |
| 1 | Lost Island | no wonder-associated fleet any more; each wonder stage advances a fleet of your choice, free |
| 1 | Inhabited Island | 3 VP |
| 2 | Abandoned Oasis | produces glass, paper or linen |
| 2 | Iron Oasis | 2 shields |
| 2 | Windy Oasis | 2 naval strength |
| 2 | Forgotten Oasis | may choose to sit out each naval conflict |
| 2 | Ancient Oasis | each Naval Construction costs 1 resource less (coins still paid in full) |
| 2 | Stormy Oasis | advance two different fleets 1 space each, free |
| 2 | Wonderous Oasis | +1 coin after each time you advance your fleets |
| 2 | Oasis of Statues | 5 VP |
| 2 | Pirate Oasis | 3 VP; pirates |
| 3 | Scarlet Archipelago | 3 shields |
| 3 | Breezy Archipelago | 3 naval strength |
| 3 | Torrential Archipelago | advance three different fleets 1 space each, free |
| 3 | Golden Archipelago | end: one science symbol of your choice |
| 3 | Emerald Archipelago | end: +1 science symbol (most) |
| 3 | Royal Archipelago | 7 VP |
| 3 | Lush Archipelago | end: 2 VP per commercial level |
| 3 | Rainbow Archipelago | end: choose an Age-card colour, 1 VP per card of it in your city |
| 3 | Sheltered Archipelago | end: 2 VP per Island you own (including this one) |

## Shipyards

**Track effects** (the same on all 7 boards; **R** figure, **F**). Values marked "current" count only while the fleet
is on that space. The others fire once, when the fleet arrives.

| Space | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Red: naval strength (current) | 1 | 2 | 3 | 4 | 5 | 8 |
| Yellow: commercial level (current) | 1 | 1 | 2 | 2 | 3 | 4 |
| Yellow: on arrival | – | +2 coins, tax 1 | +3 coins | +4 coins, tax 2 | +5 coins | tax 4; 2 naval strength (current) |
| Blue: VP (current) | 1 | 2 | 4 | 6 | 8 | 10 + 1 naval strength |
| Green: on arrival | – | explore L1 | – | explore L2 | – | explore L3 |

**Naval Construction cost pattern**: the cost of moving *to* each space. It holds on every board; **F** gives it,
and the board in **R** matches it exactly.

| To space | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| red / yellow / blue | 1 brown | 1 grey | 2 brown | 1 brown + 1 grey | 2 brown | 2 grey |
| green | 1 brown | 2 coins | 2 brown | 3 coins | 2 brown | 4 coins |

**The one board known in full** (**R** p. 2 figure; its Wonder symbol is on **red**):

| To space | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| red | wood | paper | stone, bricks | stone, glass | wood, ore | glass, linen |
| yellow | stone | linen | wood, ore | bricks, paper | stone, bricks | glass, paper |
| blue | bricks | glass | stone, ore | ore, linen | wood, bricks | paper, linen |
| green | ore | 2 coins | wood, bricks | 3 coins | stone, ore | 4 coins |

This matches both cost examples in **R**: the blue cost is 1 clay, and the red cost (the wonder fleet) is 1 wood.

**Missing**: the other 6 boards' concrete resources and their Wonder-symbol tracks. No source found has them, either
as text or as a readable image: the wiki's `Shipyard.png` has no icons, and the EN01 rulebook link serves an HTML page.
Until they are transcribed from the physical boards, two stand-ins are possible: use the known board for every seat,
or generate pattern-conforming variants with the Wonder symbol spread over the four tracks, clearly labelled as
unofficial.

## Siracusa (Armada only; starts with glass; stages in any order: **F**, the board art shows a stage marker on every stage)

| Side | Stage | Cost | Effect |
|---|---|---|---|
| A (Day) | 1 | 2 stone, ore | advance a fleet of your choice, free; 1 naval strength |
| A | 2 | 2 ore, linen | advance a fleet of your choice, free; 3 VP |
| A | 3 | 2 wood, paper | advance a fleet of your choice, free; 1 shield |
| B (Night) | 1 | ore, bricks | +6 coins; pirates (others lose coins = their commercial level) |
| B | 2 | 2 ore, wood | advance two different fleets 1 space each, free |
| B | 3 | 2 wood, paper | 1 shield; 1 naval strength |

The images in **S** use the current edition's Day/Night art. If the repo's base boards follow the first edition's A/B
layout, check that Siracusa's costs are the same in both printings.

## Discrepancies and confidence

* ⚠ **Harbor Master's Office / Naval Headquarters**: the wiki card list says Age II. The wiki's own shipyard page,
  the 8-cards-per-Age count and the VP progression (0/0/2/3/4/5) all say **Age III**, which the draft uses.
* ⚠ **Emporiums**: the wiki gives both as "from your right neighbour". The effects sheet describes them with "the
  designated player" next to the two-seats-away symbols, and "Western" suggests left. The draft uses
  `far_trade` with `right2` / `left2`. Check this against a card image.
* ⚠ **Pier's boarding target**: the wiki says left-of-left, the same as Ballista. The box has 2 Boarding tokens, so
  the official split may differ.
* **Wonderous Oasis**: level 2 in the island list and level 3 in a side table. The draft uses **2**, which keeps
  9 islands per level.
* **Blue cards' fleet move**: "a fleet of your choice" (**F**). **E** also defines a "fleet of this colour" symbol,
  which only Topaz and Amber Island use according to **F**.
* **Track values**: the yellow space-6 tax (4) comes from **F**, because the figure is too small to read it. The
  other values agree between **R** and **F**.
* **Resource icons** in the rulebook figure were told apart by colour (wood olive, ore dark, stone grey, clay orange,
  glass light blue, paper tan, linen pink). The wood / ore pairs are the least certain.
* **Hand sizes, token values, tie rules, purple cap, Cities debt**: from **R** (high confidence).
