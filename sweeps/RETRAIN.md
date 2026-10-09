# Retraining under the October 2026 rules

**Why.** Three engine bugs were fixed (PRs #28, #29): neighbours may only buy the board's resource
and brown / grey production, cards used for a wonder stage stay under the board, and discarded
leaders return to the box instead of the discard pile. Every model so far, the leader value table
and the scripted agents' leader choices predated these fixes.

**What was done.**

1. **Leader values** re-estimated under the new rules (`python -m wonders.agents.leader_values`,
   40,000 games). The top leaders are now Caesar (+2.35 VP), Midas (+2.32) and Amytis (+2.19).
2. **Fine-tuning** (`sweeps/retrain.json`): two continuations of the bundled `s5-anneal`, 10M PPO
   steps each with the `s5-anneal` recipe (tiered opponent mix, anchors, linear learning-rate
   decay from 1e-4 to 1e-5), seeds 70 (`r1-a`) and 71 (`r1-b`). Each took ~8,900 s
   (~1,120 steps/s on 6 cores + RTX 3060).

## Results (all under the new rules)

**Head-to-head** — the three models at the same tables with heuristic fillers, every expansion
combination and 3–7 players, random seating, 8,000 games per pair:

| pair | A ranked above B (95% CI) |
|---|---|
| `r1-a` vs `s5-anneal` | **52.2%** (51.1–53.3) |
| `r1-b` vs `s5-anneal` | **51.3%** (50.2–52.4) |
| `r1-a` vs `r1-b` | 51.8% (50.7–52.9) |

By expansion combination (`r1-a` vs `s5-anneal`): base 51.8%, Leaders 52.1%, Cities 50.5%,
Leaders + Cities **54.4%** (52.2–56.5) — the gain is largest where the rules changes matter most
(leaders no longer reach the discard pile; trading changes compound with Cities' coin effects).

**Against the strongest scripted agents** — each model with `adaptive` and `board`, heuristic
fillers, every expansion combination and 3–7 players, 4,000 games:

| model | ranked above `adaptive` | ranked above `board` |
|---|---|---|
| `r1-a` | **86.9%** (85.8–87.9) | **88.3%** (87.3–89.3) |
| `r1-b` | 86.6% (85.5–87.6) | 87.8% (86.8–88.8) |
| `s5-anneal` | 84.5% (83.3–85.6) | 86.1% (84.9–87.1) |

## Conclusions

* Both seeds improve on `s5-anneal` under the new rules, by about 1–2 points head-to-head and
  2–2.5 points against the scripted agents. The pre-fix model had not lost much (it still ranked
  above `adaptive` in 84.5% of games), but it misjudged trading and the discard pile.
* **`r1-a` is bundled** as `wonders/ml/pretrained/ppo_universal_v2.pt`, stamped with rules
  version 2 (`wonders.ruleset.RULES_VERSION`): it finishes first in 76% of games against
  `adaptive`, `board` and heuristic agents (even share 22%).
* Next (ROADMAP item 2): re-measure network-guided search under the new rules on top of `r1-a`, then
  the first expert-iteration round (`wonders train expert`).
