# Adaptive scripted agents

`wonders/agents/adaptive.py` adds two scripted agents on top of `BoardAwareAgent` (and its
data-driven leader play): **`adaptive`** and **`rival`**. They exist to make the scripted part
of the PPO opponent pool (and a possible imitation teacher) stronger and more varied.

## Rules

| rule | what it does | weight(s) |
|---|---|---|
| Contestable military fronts | extra shields are worth the change in *expected* conflict tokens vs each neighbour's projected shields at every remaining conflict (both sides projected with a smoothed pace; later conflicts discounted; unbuilt stage shields count) | `front`, `front_noise`, `front_discount`, `front_prior` |
| Phase awareness | production / trade value decays to `production_floor` by the last turn; chains count only if a later Age remains | `production_floor` |
| Science commitment | a symbol is worth more for a committed collector with turns left and little green competition (unbuilt stage science counts) | `science_potential` |
| Wheel awareness | while the hand will come back (more cards than players), an uncontested card can wait | `wheel` |
| Combos | what a card adds to payoffs the city owns (per-colour VP, colour sets, build income, incl. leaders in hand), projected future units of counting cards, cost of feeding a neighbour's guild, stage counters on the wonder action | `combo`, `combo_future`, `combo_hand`, `feed`, `stage_odds`, `pace_prior` |
| Research rules | military by player count, Age III upstream-front caution, science stop at two sets / no late entry, raw-material soft cap, unique-resource preference, coin hoarding | `np_military`, `upstream_front`, `science_rules`, `raw_soft_cap`, `raw_penalty`, `unique_resource`, `coin_hoard`, `coin_hoard_factor` |
| `rival` only | deny cards to the score leader when they receive the hand next; weigh the front against them more | `rival_deny`, `rival_front` |

Combos read the new `Effect.counts()` hook — what an effect pays per unit (public card text,
exposed as `counts` in effect summaries; the engine never reads it, encoders ignore it).

## Results (random seating, heuristic fillers, all 20 cells, 150 games per cell = 3,000 games per pair)

| comparison | A ranked above B |
|---|---|
| adaptive (rules, no combos) vs board | **55.3%** (53.5–57.1) |
| adaptive (default: + half combos + research rules) vs board | **54.9%** (53.1–56.7) |
| adaptive with full-strength combos vs no combos | 48.1% (46.3–49.9) |
| combo components (enabler / future / half / stages only) vs no combos | 48.9–49.9% |
| default adaptive vs adaptive without combos and research rules | 50.4% (48.6–52.2) |
| leave-one-out of each research rule vs the same baseline | 49.7–50.8% |
| rival (first version) vs board / vs adaptive | 52.3% / 47.5% |

**Reading.** The adaptive rules — chiefly the projected military fronts — carry a solid gain
over `board` (the strongest scripted agent before). Combos at full strength slightly hurt
(over-crediting counting cards); at half strength they and the research rules are neutral
against this field (within ±1 point, the resolution of 3,000 games is ~2 points). They are kept
as defaults because they cost nothing, encode documented good practice and diversify the
opponent pool. Many research rules target human opponents (denial, arms races, contested
science), which the scripted field does not reproduce; they may matter more against trained
models — untested.

## Strategy research (summary)

A web research pass collected ~50 rules. The two strongest sources:

* Bettker et al., SBGames 2021 — 4,504 three-player Board Game Arena games of top players:
  https://www.sbgames.org/proceedings2021/ComputacaoFull/216981.pdf (shields → net military VP
  curve, sweet spot ~7 shields; winners' archetypes: science 28% / military+civil 72%; above
  third-quartile raw materials → 69% loss rate; above-average final treasury → 64% loss rate;
  board win rates, e.g. Rhodes B 41%, Halicarnassus B 39%).
* "Some complex strategies", BoardGameGeek: https://boardgamegeek.com/thread/691370 (card
  distribution and resource scarcity by player count, science symbol counts per Age, wonder- and
  guild-specific play).

Implemented: player-count military, upstream caution, science thresholds, raw-material cap,
unique resources, coin hoarding. Not yet: receiver-aware valuation of Arena / Haven / guilds,
per-symbol remaining-copy tables by player count, guild resource planning, the per-Age
"bury a weak hand" benchmark (≈ 3 / 5 / 7 VP), card counting of unseen cards, Cities-specific
diplomacy rules.
