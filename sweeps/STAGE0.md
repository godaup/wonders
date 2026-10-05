# Training — Stage 0: yardstick and Leaders diagnostics

> **Correction (Stage 1, `sweeps/STAGE1.md`):** the head-to-heads and ablations below were seated
> by rotation, which favours the first-listed model by ~7 points between trained models. Re-run
> with random seating: **Finding 3 does not hold** — all v1/v2 pairs and seed pairs are 48–51%;
> Finding 1 stands (policy above cost proxy 65.6%, above random 61.9% for v2-s1; the Leaders
> re-eval numbers are single-model and unaffected); Finding 2 softens to "about random"
> (51.3% / 48.9%).

**Goal:** before more training, get a metric that still separates models once they beat the
scripted agents easily, and find out why Leaders looked weak (v2 eval: Leaders ≈ 3.1 win × n
vs ≈ 3.9–4.1 elsewhere). Models: the universal 256×2 PPO runs `ppo-C-256x2-s{0,1}` (v1
encoder) and `ppo-C2-256x2-s{0,1}` (v2 encoder) from `runs/main`.

Tools added in this stage: `wonders train h2h` (models at the same tables, heuristic filler
seats, rotating seats; "A above B" = share of games where A ranked higher) and
`wonders train leaders` (teacher agreement per decision kind + Leaders ablations).

## Finding 1 — the network never played its leaders outside training (bug, fixed)

`NeuralAgent` inherited `Agent.act`, which routes the Leaders draft and recruitment to the
default hooks: keep / recruit the **costliest** leader. During PPO the learner seat acts
through the network directly, so the policy *learned* leader decisions — but every bench,
eval, `wonders play`, `wonders gui` game and league opponent replaced them with the cost
proxy. `NeuralAgent` now plays every decision with the network (`leader_policy=True`,
default; `False` reproduces the old behaviour).

Same model, own leader policy vs cost proxy vs random leader picks (Leaders + Leaders+Cities
cells, 3–7 players, 300 games per cell):

| model | policy above cost proxy | policy above random | cost proxy above random |
|---|---|---|---|
| v2-s1 | **68.5%** (66.8–70.2) | 62.6% (60.9–64.3) | 43.3% (41.5–45.1) |
| v1-s0 | **69.6%** (68.0–71.3) | 64.9% (63.1–66.6) | 43.2% (41.5–45.0) |

Scripted eval on the Leaders cells (same seed / 400 games per cell as the sweep eval), before
→ after the fix:

| run | Leaders cells win × n |
|---|---|
| v1-s0 | 3.267 → **4.163** |
| v1-s1 | 3.316 → 3.848 |
| v2-s0 | 3.246 → 4.122 |
| v2-s1 | 3.314 → 4.002 |

**Leaders was never the weak spot — the evaluation was.** All earlier Leaders numbers
(`runs/main/*/eval.json`) understate the models by 0.5–0.9 win × n.

## Finding 2 — the scripted agents' leader play is random or worse

No scripted agent overrides `draft_leader` / `recruit`, so all of them (incl. the imitation
teacher `board`) use the cost proxy. `board` against itself with random leader picks:

| pair | A above B |
|---|---|
| board vs board[random draft] | 51.2% (49.4–53.0) — no better than random |
| board vs board[random draft + recruit] | 46.4% (44.7–48.2) — *worse* than random |

So imitation clones meaningless leader picks, and PPO trained against opponents whose leader
play is poor. Greedy-policy agreement with the teacher (v2-s1 / v1-s0): play 49.7% / 47.2%,
draft 57.7% / 49.3%, recruit 42.9% / 37.6% — PPO has moved well away from the teacher.

## Finding 3 — seed variance dwarfs the encoder difference

Head-to-head, all four models at the same 4–7 player tables (3-player cells skipped: four
models), 400 games per cell, 6,400 games per pair:

| pair | A above B (95% CI) |
|---|---|
| v1-s0 vs v1-s1 | **60.3%** (59.1–61.5) |
| v2-s0 vs v2-s1 | 54.0% (52.7–55.2) |
| v1-s0 vs v2-s0 | 51.3% (50.1–52.6) |
| v1-s0 vs v2-s1 | 55.2% (54.0–56.5) |
| v1-s1 vs v2-s0 | 41.0% (39.8–42.2) |
| v1-s1 vs v2-s1 | 44.9% (43.6–46.1) |

Order: v1-s0 > v2-s0 > v2-s1 > v1-s1. Averaged over the four cross-encoder pairs v2 ranks
above v1 in 51.9% of games — a small edge, far below the 60/40 gap between two v1 seeds that
the scripted eval scored identically (3.61 vs 3.60). (Measured with the leader fix in place.)

## Consequences for the next stages

* **Primary metric is now head-to-head** (`wonders train h2h`); the scripted eval stays as a
  regression guard. Every comparison needs **≥ 2 seeds per arm**.
* **Stage 1** (league vs control) runs as planned, but with two seeds per arm, and league
  snapshots now play leaders properly.
* **Stage 2 changes from "fix Leaders data" to "fix the scripted leader play"**: give the
  heuristic/board agents value-based `draft_leader` / `recruit` hooks (or drop draft/recruit
  from the imitation targets), so the teacher and the opponent pool stop teaching poor leader
  play.
* Everything was measured with `~/.venvs/wonders-train`-equivalent code at `stage0-eval`;
  raw results: `runs/stage0/` (`h2h.json`, `leaders-*.json`, `reeval-leaders-*.md`).
