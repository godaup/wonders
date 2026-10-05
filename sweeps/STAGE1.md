# Training — Stage 1: self-play league vs more scripted training

**Question:** does continuing PPO against the learner's own snapshots (the *league*) beat simply
training longer against the scripted pool? Spec: `sweeps/stage1.json`. Both universal v2 runs
(`ppo-C2-256x2-s{0,1}`) continue for 10M learner steps, once against the 7 scripted agents only
(**ctrl**) and once with the learner's snapshots added (every 50 iterations, 6 kept; **league**).
Within a pair the PPO seed is the same, so the arms differ only in the opponent pool.

| run | init | time (incl. eval) | steps/s |
|---|---|---|---|
| s1-ctrl-a | ppo-C2-256x2-s0 | 4,868 s | ~2,100 |
| s1-league-a | ppo-C2-256x2-s0 | 7,828 s | ~1,400 |
| s1-ctrl-b | ppo-C2-256x2-s1 | 5,000 s | ~2,100 |
| s1-league-b | ppo-C2-256x2-s1 | 7,543 s | ~1,450 |

The league runs are ~1.5× slower: snapshot opponents run network inference in the env workers.

## Measurement note — head-to-head seating

The first head-to-heads used `run_bench`'s rotation, which keeps the models' relative order: the
first-listed model always picks from a hand before the next one (Ages I and III). Between two
trained models that is worth ~7 points: league-a ranked above ctrl-a in 43.7% or 57.7% of games
depending only on the listing order, and 50.4% / 50.2% with random seating. Identical heuristic
agents show no such effect, so it only appears between trained policies. Head-to-heads now seat
every game at random (`seating='shuffle'`, the default). All numbers below use it; the rotation
results are kept as `runs/stage1/*-rotate.*`.

## Results (random seating, 400 games per cell)

Each pair with its starting checkpoint, all 20 cells (8,000 games per pair):

| pair | A above B (95% CI) |
|---|---|
| league-a vs ctrl-a | **54.9%** (53.8–56.0) |
| league-a vs init-a | **55.9%** (54.8–56.9) |
| ctrl-a vs init-a | 50.6% (49.5–51.7) |
| league-b vs ctrl-b | **54.7%** (53.6–55.8) |
| league-b vs init-b | **55.0%** (53.9–56.1) |
| ctrl-b vs init-b | 50.6% (49.5–51.7) |

All four new models together, 4–7 players (6,400 games per pair): league-a above ctrl-a 52.9%,
league-b above ctrl-b 54.4%, league-b above league-a 53.1%, league-a above ctrl-b 52.4%,
league-b above ctrl-a 56.2%, ctrl-a vs ctrl-b 50.1%.

Scripted eval (regression guard, pair a): ctrl-a 4.126 vs league-a 4.071 win × n
(−0.055 ± 0.023), mostly on the Leaders cells.

## Interpretation

* **Gate passed:** the league beats the control in both pairs (~+5 points head-to-head), while
  10M more steps against the scripted pool gained nothing (ctrl ≈ init).
* The league's edge shows at tables with other trained players. With two models and heuristic
  fillers only, league-a and ctrl-a tied (50.3%), and the league is slightly worse at exploiting
  the scripted agents (scripted eval −0.055). Self-play teaches play against strong opponents —
  which is what matters for a model meant to beat its predecessors or humans.
* **The league becomes the default** for the next stages. Cost: ~1.5× wall time per step.

## Corrections to Stage 0 (re-measured with random seating)

* v1 vs v2 and seed vs seed: every pair is 48–51% (`runs/stage0/h2h.*`). The encoders and the
  seeds are equivalent head-to-head; the 60/40 "seed variance" was the seating bias.
* Leader policy vs cost proxy (v2-s1): 65.6% (63.9–67.3), vs random 61.9% — the leader-routing
  fix stands. The board agent's leader play is about random (51.3% / 48.9%), not worse.

## Next

Stage 2 — fix the scripted leader play (value-based `draft_leader` / `recruit` hooks for the
heuristic/board agents, or drop draft/recruit from the imitation targets), continuing from the
league runs. Best current checkpoints: `s1-league-b`, then `s1-league-a`.
