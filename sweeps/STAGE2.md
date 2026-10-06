# Training — Stage 2: better scripted leader play in the opponent pool

**Question:** Stage 0 found that every scripted agent played leaders by the cost proxy (keep /
recruit the costliest leader — about as good as random). Does a pool whose scripted agents play
leaders well make the trained policy stronger?

## The scripted leader fix

`wonders/agents/leader_values.py` estimates what recruiting each leader is worth: 40,000 games
of board agents with random leader play (Leaders and Leaders+Cities, 3–7 players), final score
margin regressed on the recruited leaders with a per-table intercept (standard errors
0.10–0.13). Best: Midas +2.8, Caesar +2.4, Amytis +2.3, Cleopatra +1.7; worst: Imhotep −2.1,
Aristotle −2.0, Archimedes −1.7, Maecenas −1.6 (relative to using the recruitment for a stage /
3 coins). `HeuristicAgent` (and so `board`, `board-focus`) now drafts the most valuable leader,
recruits the best affordable one unless a stage or 3 coins is worth more, and uses the values
for Rome's extra recruit and the Courtesans' copy.

Held-out seeds, random seating, Leaders cells (3,000 games per pair): `board` ranks above
`board[cost]` (old behaviour) in **62.2%** and above `board[random]` in **57.9%** of games —
close to the network's own leader policy (65.6% above its cost-proxy twin).

## Training runs (`sweeps/stage2.json`)

Both Stage 1 league runs continue for 10M steps with the league on; the arms differ only in the
scripted opponents' leader play (same PPO seed within a pair):

| run | init | scripted pool | time |
|---|---|---|---|
| s2-new-a | s1-league-a | data-driven leaders | 7,694 s |
| s2-old-a | s1-league-a | `heuristic[cost]`, `board[cost]`, `board-focus[cost]` | 7,306 s |
| s2-new-b | s1-league-b | data-driven leaders | 7,280 s |
| s2-old-b | s1-league-b | cost proxy | 7,279 s |

(The other scripted agents — greedy, builder, civic, scientist — are not heuristic-based and the
same in both arms.)

## Results (random seating, 400 games per cell)

| pair (all 20 cells, 8,000 games) | A above B (95% CI) |
|---|---|
| new-a vs old-a | 50.4% (49.3–51.5) |
| new-b vs old-b | 50.0% (48.9–51.1) |
| new-a vs init-a | 51.8% (50.7–52.8) |
| old-a vs init-a | 52.2% (51.1–53.3) |
| new-b vs init-b | 51.2% (50.1–52.3) |
| old-b vs init-b | 51.7% (50.6–52.8) |

All four together (4–7 players): new-a vs old-a 50.7%, new-b vs old-b 50.5%, every other pair
49.0–50.8%. Leaders cells alone: new vs old 49.5% (a) / 51.1% (b); both arms gained most over
their init in the Leaders cells (~53–54%).

## Interpretation

* **Gate not passed:** better scripted leader play in the pool does not change the trained
  policy — the network already plays leaders far better than any scripted agent, so its
  opponents' draft quality barely affects what it learns.
* Continued league training still helps, but less: ~+2 points per 10M steps (Stage 1: ~+5),
  most of it in Leaders games.
* The leader values stay: they make the scripted agents much stronger (62% vs their old self),
  which matters for the imitation teacher, the GUI opponents and benchmarks.
* Best checkpoints now: the four `s2-*` runs are equivalent; any of them is ~+2 points over
  `s1-league-*` and ~+7 over the Stage 0 models.

## Next

Diminishing returns from the same recipe suggest changing the opponent mix or the model rather
than training longer: long-lived league anchors and a snapshot-heavier pool (the mix experiment
proposed after Stage 1), adding the new `adaptive` agent to the scripted pool, and/or Stage 3
(capacity). Stage 4 (promote a universal checkpoint) can use any `s2-*` run.
