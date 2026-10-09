# Expert iteration, round 1: distilling flat search into the network

**Question.** Network-guided search plays clearly better than the network it searches with. Can
the network learn that improvement — distil the search's choices into the policy (expert
iteration), as AlphaZero-style systems do?

**Teacher check (current rules).** `SearchAgent` (3 turns ahead, 16 sampled worlds, the policy's
top 4 candidates) against the plain bundled network `r1-a`, same tables, heuristic fillers, 800
games over 8 cells: search ranks above `r1-a` in **56.9%** (53.4–60.3) of games (base 56%, Leaders
53.5%, Cities 61%, Leaders + Cities 57%). The teacher is valid.

**Collection.** `wonders train expert` with `r1-a`: 6,000 games over every expansion combination
and 3–7 players, one searching seat per game (rotating), the plain network in the other seats;
140,704 recorded decisions (~23 per game), 4.4 h on 11 processes. The network already plays the
search's top move in **72%** of the decisions.

## Distillation variants

Each variant fine-tunes `r1-a` on the same data (soft cross-entropy against the targets plus value
regression), then plays `r1-a` at the same tables (8,000 games, every expansion combination and
3–7 players, random seating):

| variant | targets | training | ranked above `r1-a` (95% CI) |
|---|---|---|---|
| C | softmax(value / 2) | lr 1e-4, 3 epochs | 35.4% |
| A | softmax(value / 1) | lr 1e-4, 3 epochs | 40.9% |
| D | softmax(value / 1) | lr 3e-5, 3 epochs | 42.5% |
| B | softmax(value / 0.5) | lr 1e-4, 3 epochs | 43.2% |
| E | softmax(value / 0.25) | lr 1e-4, 3 epochs | 44.4% |
| F | softmax(value / 0.1) | lr 1e-4, 3 epochs | 44.9% |
| H | hard targets, only decisions where the best candidate leads by ≥ 1 VP (53k of 141k) | lr 1e-4, 3 epochs | 49.0% (47.9–50.1) |
| G | softmax(value / 0.1) | lr 3e-5, 1 epoch | 50.1% (49.0–51.2) |

**PPO polish.** H continued for 3M PPO steps (lr 3e-5 → 3e-6, the `r1-a` recipe), against a control
of `r1-a` with the identical polish: polished H vs control **49.7%** (48.6–50.8), vs `r1-a` 49.3%;
control vs `r1-a` 50.0%.

## Findings

* **The softer the targets, the worse the network.** At temperature ≥ 0.5 the candidates' values
  (within ~1 VP of each other, with roughly ±1 VP of sampling noise from 16 worlds) give nearly
  uniform targets over the top 4 moves; the distilled policy becomes flatter (entropy 0.25 vs 0.16)
  and its greedy move arbitrary among them. Held-out agreement with the search's best move *fell*
  from 0.72 to ~0.64 during training.
* **Sharp targets are at best neutral.** Near-hard targets, confident-only hard targets and a PPO
  polish all land at 49–50%: the search's corrections, a minority of decisions mixed with its own
  noise, do not make the network stronger.
* **So the search's advantage does not live in the policy's reach.** It comes from per-position
  lookahead in sampled worlds; flat search with this noise level is not a usable teacher.

## Conclusion and next steps

Flat-search expert iteration is shelved; `r1-a` stays bundled. The remaining routes to a stronger
network are a better teacher — **information-set MCTS** (or decoupled UCT for the simultaneous
moves), whose visit counts give sharp, well-founded targets — and **opponent modelling** for the
determinisation (sampling hidden hands from a learned belief instead of uniformly), both in
`ROADMAP.md`. Search itself stays useful as a stronger player (difficulty levels, post-game
analysis, puzzles).

The pipeline (`wonders/ml/expert.py`, `wonders train expert`) stays in place for a future teacher:
it stores the candidate scores, so new targets can be derived from one collection.
