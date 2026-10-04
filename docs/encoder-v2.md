# Observation encoder v2 (`table-v2`)

## Why a new encoder

The training sweep showed that **network capacity is not what limits the learned agent**:

| universal model (20 cells) | mean win × fair share |
|---|---|
| 256×2 MLP, seed 0 | 3.61 |
| 256×2 MLP, seed 1 | 3.60 |
| 512×3 MLP (3× the parameters), seed 0 | 3.55 |

Imitation of the deterministic board-aware teachers also plateaued at ~74 % action agreement for
both sizes. A network that saw everything the teachers see could, in principle, match them; the
plateau points at **missing information**. Encoder v1 (`card-v1`, `CardEncoder`, 851 inputs)
encodes only the viewer and its two neighbours with coarse features.

**v2 encodes everything a seat may legally observe** (`TableEncoder`, 3,020 inputs with the
universal vocabulary). Hidden information — opponents' hands and leader hands, the identity of
cards others buried or discarded — never enters (tested).

## What is observable (and what v1 missed)

| information | v1 | v2 |
|---|---|---|
| own / neighbours' coins, shields, stages, score, science, colour counts | ✓ | ✓ |
| built cards of own city and both neighbours | ✓ | ✓ |
| **all other cities** (5–7 player games: Diplomats/Architects guilds, "all other players" effects) | – | ✓ |
| **score breakdown** per city (9 categories, incl. leaders / cities) | – | ✓ |
| **military tokens by value** (1/3/5) and defeats (Strategists, Tomyris, Memorial, Sepulcher…) | sum only | ✓ |
| **debt** and **diplomacy** tokens (Cities) | – | ✓ |
| **science wilds** | – | ✓ |
| **production** per city: fixed units and choice slots per resource, own-use slots | – | ✓ |
| **wonder identity and side** per city, **next stage cost** per city | – | ✓ |
| Courtesans' Guild copies | – | ✓ |
| hand: buildable, total coin cost | cost only | ✓ + trade part, chain-free, copies held |
| **own leader hand** (private) | – | ✓ |
| **cards seen earlier this Age** that moved on (card counting) | – | ✓ |
| **cards the viewer itself discarded / buried** (it knows them) | – | ✓ |
| **trade prices** per neighbour and resource | – | ✓ |
| **own whole wonder board**: per stage cost, effects (incl. powers), built flag | next stage affordable only | ✓ |
| player count, expansions in play, pass direction | – | ✓ |
| Age, turn, hand size, discard count, free builds left, decision kind | ✓ | ✓ |
| visible moves per seat (history) | – (FlatEncoder only) | ✓ |

## Layout

Seats are **relative to the viewer**: slot 0 is the viewer, slot *k* the city *k* seats clockwise
(slot 1 = left neighbour, slot *n−1* = right neighbour), padded with zero slots to 7, so the size
does not depend on the player count and "left neighbour" is always in the same place.

| section | size | content |
|---|---|---|
| `city0` … `city6` | 7 × 229 | present / left / right flags · coins, shields, stages built/total, score · score breakdown (9) · victory tokens 1/3/5, defeats · debt, diplomacy · science symbols (3), wilds · colour counts (9) · production: fixed units (7), choice slots (7), own-use slots · wonder one-hot (10), side B · next stage cost (7 resources, coins), has next stage · Courtesans copies · built cards (157, multi-hot) |
| `hand` | 5 × 157 | per card name: copies in hand · buildable · coin cost · trade part · chain-free |
| `private` | 583 | own leader hand (157) · seen cards this Age (157, counts) · own discarded/buried (157, counts) · buy prices left/right per resource (14) · free builds left, stage affordable · own board: 4 stages × (cost 7+1, effect features 15, built) |
| `context` | 21 | Age one-hot · turn, hand size, discard count · player count one-hot (3–7) · leaders / cities in play · pass left · decision kind one-hot (7) |
| `history` | 7 × 4 | visible moves per seat slot: build / wonder / discard / free |

`TableEncoder.sections()` returns `(name, offset, length)` for each block. Values are roughly
unit-scaled (coins / 10, score / 50, …); `PolicyValueNet` additionally standardizes inputs with
statistics fitted on the imitation data.

Effect features (per wonder stage): VP, shields, coins, science symbol, science wild, production
slots, the eight power hooks (`plays_last_card`, `free_builds_per_age`, `discard_builds`,
`guild_copies`, `leader_draws`, `extra_recruits`, `leader_copies`, `science_copies`) and diplomacy.

## Supporting observation fields

v2 needed the `Observation` to carry the public data the engine already had:

* `CityView`: `tokens`, `debt`, `diplomacy`, `science_wilds`, `production`, `private_slots`,
  `score_breakdown`, `copied_leaders`, `next_stage_cost`;
* `CardView`: `chain_free`, `bank_coins`, `trade_coins`;
* `Observation`: `leaders_hand`, `seen_cards`, `buy_prices`, `expansions`, `pass_left`.

`seen_cards` comes from `ObservationBuilder`, which remembers each seat's own regular-turn hands
per Age (only what that seat saw). Score breakdowns are cached per engine tick
(`Game.cached_breakdowns`), like the totals before.

## Versioning and compatibility

* `Vocab.encoder_kind` selects `'card-v1'` (default) or `'table-v2'`; `universal_vocab('table-v2')`.
* Checkpoints store `encoder`, `decisions` and `wonders` (`Vocab.meta()`); `NeuralAgent` rebuilds
  the right encoder. Old checkpoints (no `encoder` key) load as v1 — the bundled model and the
  running sweep's models keep working.
* `PPOConfig.encoder` / sweep runs' `"encoder"` key choose the encoder; an init checkpoint must
  match the PPO run's encoder (checked).

## Cost

Encoding v2 takes ~75 µs per observation vs ~15 µs for v1 — negligible, since only the learner's
observation is encoded. Building the richer observations (for every acting seat, including
scripted opponents) costs roughly 20–25 % more simulation time in a first single-process
measurement (taken while a sweep was loading the machine; to be re-measured).

## Next steps

1. **Head-to-head**: train a universal 256×2 model with v2 on the same budget as `ppo-C-256x2-s0`
   (imitation → 20M PPO steps) and compare per cell (`sweeps/encoder_v2.json`).
2. **Structured model**: the seven `city` blocks share one layout — embed them with shared weights
   (and pool / attend over opponents), plus a per-card embedding for the hand, instead of one flat
   MLP over 3,020 inputs. `sections()` provides the slices.
3. **Imitation ceiling**: check whether teacher agreement rises above ~74 % with v2 — a direct test
   of whether the missing information was the limit.
