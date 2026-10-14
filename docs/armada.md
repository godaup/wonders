# Armada

Armada adds a Shipyard board per player with four fleets, 24 Age cards, 27 islands, naval conflicts, Boarding
and the Siracusa board. Both editions are playable: `armada2` (second edition, 2020; with `base2`) and `armada`
(first edition, 2018; with `base`). The component data and its sources are in [`armada-data.md`](armada-data.md).

```
wonders play --edition 2 -e armada -a board,heuristic,greedy
wonders play -e armada -e cities          # first edition
```

## Rules as implemented

* **Hands**: each Age gets as many random Armada cards as players (`"deck": "draw"`; the five 4+ cards leave a
  3-player game), so hands are 8 cards, 9 with Cities (`decks.deck_size` counts them).
* **Shipyards** (`Content.shipyards`, `Player.shipyard`): four tracks of six spaces. A space's status effects
  (naval strength, commercial level, blue VP) count while the fleet stands on it (`Player.fleet_effects`, part of
  `iter_effects`); its other effects (coins, Taxes, explorations) fire once on arrival (`Game._advance_fleet`
  activates the space like a card). Each seat is dealt a random board at setup (`record.with_shipyards`), stored
  in `PlayerSetup.shipyard`.
* **Naval Construction** (`Action.naval`, part of equality): building a red, yellow, blue or green card, or a
  wonder stage, may also advance that colour's fleet or the wonder's fleet. Both are paid together in one
  payment (`Game._card_payment(naval=True)`, `_stage_payment(naval=True)`); a card that is free (chain, Olympia)
  pays only the Naval Construction; the card's effects apply first. A card built from the discard pile never
  has one. `legal_actions` lists each naval variant right after its action.
* **Selling**: a discard with `naval` advances the yellow fleet instead of taking 3 coins.
* **Taxes** (`tax` hook): collected per turn in `_flush_pending`. Only the highest applies, to every player but
  those who raised it, minus their commercial level. **Pirates** (`pirates`): every other player loses as many
  coins as their commercial level. Foggy Island protects from both. With Cities, coins a player cannot pay
  become debt.
* **Islands**: explorations (green fleet spaces) are resolved after the turn (`Game._naval_steps`, decision
  `EXPLORE`): a lone explorer draws 4 islands of the level, several share the whole deck equally; each keeps
  one, the rest go back and the deck is shuffled. Helmsman's Rest and the like put the top island into play at
  once (`island_draw`). Island production is for its owner only.
* **Free fleet advances**: a fixed fleet (Topaz / Amber Island) moves at once; "a fleet of your choice" (blue
  Armada cards, Siracusa, Stormy Oasis, Torrential Archipelago, Lost Island) is the decision `ADVANCE_FLEET`,
  answered with `Action.build` of a fleet card (`Red Fleet` ...). Several advances from one effect go to
  different fleets.
* **Boarding** (`boarding`): the target fights one more military conflict with the attacker at the end of the
  Age; diplomacy does not stop it. In the second edition every Boarding is a conflict of its own; the first
  edition's FAQ allows one conflict per pair of players and Age (`rules.boarding_once`).
* **Naval conflicts** (`Game._resolve_naval`, after the military conflicts): the weakest take −1/−2/−3, the
  strongest 3/5/7, the second 1/3/5. Players tied at a rank take the next rank's token and push the ranks below
  down one; nobody takes a token when all are equally strong. The first edition adds a third rank in Age III
  (+3, `rules.naval_third_points`).
* **Emporiums** (`far_trade`): buy from the player two seats away for 1 coin, once / twice per build
  (`Payment.far_left` / `far_right`; none at 3 players, where the cards are not used).
* **Scoring**: naval tokens and islands get their own columns (`ScoreBreakdown.naval` / `islands`, listed by
  `as_dict` only in Armada games); the blue fleet's points go to civilian. Each guild scores at most 10
  (`rules.guild_vp_cap`). Science above three of a kind already scores `n²`.
* **Records**: a naval move has a `+` after its code (`B+`, `W+`, `F+`, `D+`); every island drawn is kept in
  `GameRecord.islands`, so replays deal the same islands without the shuffle.

## Agents

Every agent gets Armada decisions through `Agent.act` (`wonders/agents/naval.py`): `naval` adds a Naval
Construction to the chosen play when the next space is worth its cost (a VP estimate per track), `explore`
keeps the most valuable island and `advance_fleet` takes the most valuable free step. Against the same agents
that never sail, heuristic and board agents with it score +12 VP and win 86% of 4-player games (200 games,
second edition). Random agents pick naval variants like any other legal action.

The environment adds a naval block to the action space (`ActionSpace(naval=True)`, index `n_base + card * 4 +
kind`) when Armada is in play. No network is trained on Armada: the bundled one refuses Armada tables (its card
vocabulary lacks the cards), and search (`sample_world`) does not rebuild fleets and islands yet.

## Approximations

* **Shipyard boards**: only the board printed in the rulebooks (board I) is known; the others follow the
  official cost pattern with the resources rotated and a different wonder fleet (`"official": false`) until the
  printed boards are transcribed.
* **Choices made for the players**: an opponent who must give up coins always pays what it can (with Cities the
  rest becomes debt); a player with Forgotten Oasis sits out a naval conflict exactly when it would take the
  defeat token.
* **Turn order**: coin losses (Taxes, Pirates) are applied with the other end-of-turn effects, before discard
  builds (the rulebook: after them).
* **First edition**: Lost Island ("Naval Constructions through the wonder are free and of any colour") is played
  like the second edition's (each stage advances a fleet of your choice for free); Wondrous Harbor pays per fleet
  advance. The yellow coins follow the 2018 board figure (3/4/5/6; the rulebook text suggests 2/3/4/5), and
  Siracusa's stage costs are the second edition's.
* **Not implemented**: the team variant and the 8-player rules.

## Later

Armada-aware search worlds and analysis, an encoder block for fleets and islands, a model trained with Armada,
and the printed Shipyard boards.
