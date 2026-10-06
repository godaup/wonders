# Training — Stage 3: opponent mix + larger network

Stages 2 and 3 of the original plan, combined: a deliberate opponent mix, used to train a larger
network from scratch, with a same-recipe 256×2 control.

## What the evidence says about opponents

| finding | source |
|---|---|
| Training only against the scripted pool for 10M more steps gained nothing head-to-head (control ≈ its init, 50.6%) — the scripted field is saturated (the models beat it in 75–95% of games). | Stage 1 |
| Adding recent self-snapshots (~46% of opponent seats) gained ~5 points head-to-head; the gain shows at tables with other trained players, not against scripted-only tables. | Stage 1 |
| Whether the scripted opponents draft leaders well did not matter (50/50); another 10M league steps gained only ~2 points. | Stage 2 |
| Neighbour interaction between trained policies is strong (the seating artefact: picking from a hand right after a strong drafter costs ~7 points). | Stage 1 |
| The league only ever kept the newest 6 snapshots (~1.2M steps of history): nothing stops the policy from forgetting how to beat older strategies or cycling. | `ppo._snapshot` |

So the useful signal comes from *strong, trained* opponents, and the main risks are a narrow
self-play window (forgetting / cycling) and losing robustness against unusual play (humans in
`wonders gui`, scripted styles).

## The mix

Every opponent seat first draws a **tier**, then an entry of that tier uniformly
(`PPOConfig.tier_weights`; the draw happens per environment whenever the league rebuilds the envs,
every 50 iterations):

| tier | share of seats | contents | purpose |
|---|---|---|---|
| **Scripted** | 30% | `adaptive` ×3, `board` ×2, `rival`, `board-focus`, `heuristic`, `greedy`, `builder`, `civic`, `scientist`, `militarist`, `blocker` (repeats = weight; ~57% strong, ~43% stylistic) | robustness to diverse and odd play; never forget the basics; human-like styles |
| **Recent self** | 40% | the newest 6 league snapshots (one every 50 iterations ≈ 200k steps) | the main learning signal: an opponent that keeps up |
| **Anchors** | 30% | fixed: the three best previous models (`s2-new-a`, `s2-new-b`, `s2-old-b`; two lineages); plus own long-lived snapshots, one every 1,000 iterations (≈4M steps), the newest 8 | anti-forgetting / anti-cycling; ties training to the head-to-head metric (beat the previous best) |

**Curriculum for a fresh network.** For the first 50 iterations there are no snapshots yet, so
every seat is scripted. Fixed anchors join at iteration 1,000 (≈4M steps), once the new model has
grown past imitation strength; until then the 30/40 scripted/recent weights renormalise to
43/57. (Facing models that win most games from step 0 would give a weak, noisy learning signal.)

**Implementation.** `PPOConfig.tier_weights`, `anchors`, `anchors_from`, `league_anchor_every`,
`league_anchor_max` (`wonders/ml/ppo.py`, `draw_opponents`); sweep specs may name anchors as
`run:<name>` (that run's `final.pt`). Resume keeps the self-anchors (`state['self_anchors']`).
Empty `tier_weights` keeps the old uniform draw, so earlier specs reproduce.

## Training plan (`sweeps/stage3.json`)

Both arms use the same recipe, so the comparison isolates network size:

1. **Imitation** of the strongest scripted agent: tables with `adaptive` seats (the encoder-v2
   imitation tables with `board` → `adaptive`), 12,000 games, `adaptive` as teacher (now with
   data-driven leader play), **5 epochs** (v2 imitation peaked at epoch 4 of 12).
2. **PPO, 40M learner steps** with the mix above — the same total budget the `s2-*` models had
   (20M + 10M + 10M), so the 256×2 control also measures the recipe against the old one.

| run | network | params | role |
|---|---|---|---|
| `s3-big` | 1024×3 | 5.8M | the larger network |
| `s3-small` | 256×2 | 1.0M | same-recipe control |

Order: `imit-s3-big`, `s3-big`, `imit-s3-small`, `s3-small` — the big model's result arrives first.
CPU inference per decision (single thread): 0.10 ms (256×2) vs 0.74 ms (1024×3), which only
slows simulation where the big model's own snapshots sit at the table.

## Evaluation and gates

Head-to-head with random seating (400 games per cell, all cells), plus the scripted eval as a
regression guard:

* `s3-big` vs `s3-small` — **size effect** (gate: big ranks above small, ≥ 52%).
* `s3-small` vs `s2-new-a/b` — **recipe effect** at equal budget.
* `s3-big` vs `s2-new-a/b` — **overall**: the candidate for Stage 4 (promote a universal
  checkpoint) if it beats the previous best.

One seed per arm: with fair seating, seeds of the same recipe have measured 48–51% against each
other (Stage 0/1), so a single seed is informative, but a gap under ~2 points will be treated as a
tie.

## Results

*(pending)*
