# Training sweep — Phase 1 pilot results

**Goal:** choose PPO settings for the main sweep. Two cells (base 3-player, Leaders+Cities
5-player), universal vocabulary, imitation of the board-aware agents, then 500k learner steps
of PPO against a scripted pool (heuristic, board, board-focus, greedy, builder, civic,
scientist). Each model was evaluated on its cell over 600 held-out games per reference lineup.
"win × n" is the win rate relative to the fair share (1.0 = as good as the field; 3.0 = max at
3 players, 5.0 at 5 players). 95% intervals are about ±0.08 (base 3p) and ±0.2 (L+C 5p).

## Decision

**lr = 1e-4, entropy = 0.01, rollout = 64** (`python sweeps/generate.py --lr 1e-4 --entropy 0.01 --rollout 64`).

* Base 3-player does not discriminate: all nine settings land at 2.61–2.75 (87–92 % wins).
* Leaders+Cities 5-player does: lr = 3e-5 is clearly too slow (2.35–2.42 vs 2.65–2.87).
* lr = 1e-4 / entropy = 0.01 is best in the hard cell (2.87) and best averaged over both cells
  (2.76); lr = 1e-4 / entropy = 0.003 is within noise (2.74). lr = 3e-4 reaches the highest
  *training* win rates but not better evaluation — likely more fitting to the training pool —
  and the main runs are 4–40× longer, so the steadier lr is preferred.
* Rollout 128 is not better than 64 (2.75 vs 2.87, same throughput).

## Throughput (6-core i5-12400F, 24 envs / 4 per worker)

| cell | learner steps/s | run time incl. eval (500k steps) |
|---|---|---|
| base 3p | ~1,850 | ~280 s |
| Leaders+Cities 5p | ~1,250 | ~415 s |

Imitation: ~110k–130k teacher decisions in ~25 s; held-out action agreement 75 % (base 3p) and
69 % (L+C 5p). Revised main-sweep estimate: ≈ 28 h universal + 10 h per-combo + 15 h
specialists + ≈ 10 h evaluation ≈ 2.5–3 days on this machine.

## All runs

| run | eval win × n | vs board | vs heuristic | train win_rel (last 10) | steps/s | iters |
|---|---|---|---|---|---|---|
| ppo-lc-5p-entropy_coef0.01-lr0.0001 | 2.87 | 2.60 | 3.14 | 2.90 | 1249 | 325 |
| ppo-lc-5p-entropy_coef0.003-lr0.0003 | 2.82 | 2.52 | 3.12 | 3.27 | 1252 | 325 |
| ppo-lc-5p-entropy_coef0.01-lr0.0003 | 2.80 | 2.60 | 2.99 | 3.39 | 1253 | 325 |
| ppo-base-3p-entropy_coef0.003-lr0.0001 | 2.75 | 2.68 | 2.83 | 2.69 | 1852 | 325 |
| ppo-lc-5p-rollout128-entropy_coef0.01-lr0.0001 | 2.75 | 2.52 | 2.97 | 3.09 | 1257 | 162 |
| ppo-lc-5p-entropy_coef0.003-lr0.0001 | 2.73 | 2.52 | 2.95 | 3.05 | 1252 | 325 |
| ppo-lc-5p-entropy_coef0.001-lr0.0001 | 2.68 | 2.56 | 2.81 | 3.02 | 1250 | 325 |
| ppo-base-3p-entropy_coef0.001-lr0.0001 | 2.67 | 2.52 | 2.81 | 2.63 | 1840 | 325 |
| ppo-lc-5p-entropy_coef0.001-lr0.0003 | 2.65 | 2.56 | 2.75 | 3.20 | 1241 | 325 |
| ppo-base-3p-entropy_coef0.001-lr0.0003 | 2.65 | 2.58 | 2.73 | 2.63 | 1852 | 325 |
| ppo-base-3p-entropy_coef0.01-lr0.0001 | 2.65 | 2.54 | 2.76 | 2.64 | 1840 | 325 |
| ppo-base-3p-entropy_coef0.01-lr3e-05 | 2.65 | 2.52 | 2.79 | 2.59 | 1825 | 325 |
| ppo-base-3p-entropy_coef0.01-lr0.0003 | 2.63 | 2.52 | 2.75 | 2.67 | 1850 | 325 |
| ppo-base-3p-entropy_coef0.003-lr0.0003 | 2.63 | 2.56 | 2.69 | 2.66 | 1849 | 325 |
| ppo-base-3p-entropy_coef0.001-lr3e-05 | 2.62 | 2.44 | 2.81 | 2.59 | 1839 | 325 |
| ppo-base-3p-entropy_coef0.003-lr3e-05 | 2.61 | 2.45 | 2.76 | 2.53 | 1844 | 325 |
| ppo-lc-5p-entropy_coef0.01-lr3e-05 | 2.42 | 2.17 | 2.66 | 2.51 | 1250 | 325 |
| ppo-lc-5p-entropy_coef0.003-lr3e-05 | 2.40 | 2.09 | 2.70 | 2.35 | 1258 | 325 |
| ppo-lc-5p-entropy_coef0.001-lr3e-05 | 2.35 | 2.07 | 2.63 | 2.25 | 1258 | 325 |
