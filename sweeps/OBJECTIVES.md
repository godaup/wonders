# Training — beyond the plateau: learning-rate decay vs rank reward

**Question.** `s3-small` (the bundled model) stopped improving after ~20M of its 40M PPO steps:
final vs its own snapshots at 10M / 20M / 30M ranked above them in 56.8% / 51.6% / 51.3% of
games. Is that an optimisation limit (constant lr 1e-4 too noisy to settle) or an objective
mismatch (training on score margin, evaluating finishing order)?

**Runs** (`sweeps/objectives.json`): three continuations of `s3-small`, 10M steps each, the
Stage 3 opponent mix (anchors active from the start), one PPO seed, one change per arm:

| run | change |
|---|---|
| `s5-plain` | none (control) |
| `s5-anneal` | `lr_schedule='linear'`: lr decays from 1e-4 to 1e-5 over the run |
| `s5-rank` | `reward='dense_rank'`: the dense margin plus a terminal bonus of +10 VP for first place, falling linearly to −10 for last |

Each took ~8,870 s (~1,150 steps/s).

## Results

Head-to-head, all four at the same tables, random seating, 4–7 players (6,400 games per pair):

| pair | A above B (95% CI) |
|---|---|
| anneal vs plain | **52.7%** (51.5–53.9) |
| anneal vs init (`s3-small`) | **53.4%** (52.2–54.6) |
| anneal vs rank | **52.7%** (51.5–53.9) |
| rank vs plain | 50.3% (49.1–51.5) |
| plain vs init | 50.7% (49.5–51.9) |

`anneal` beats the control in every expansion combination (52.6–53.4%). Scripted eval (win × n):
plain 4.145, anneal **4.208** (+0.063 ± 0.022 vs plain), rank 4.101 (−0.044 ± 0.022), init 4.130.

## Interpretation

* **Learning-rate decay helps** (~+3 points head-to-head over the control, confirmed by the
  scripted eval): the plateau was partly an optimisation limit — a constant 1e-4 keeps the policy
  too noisy to settle. Adopt `lr_schedule='linear'` for final training phases.
* **The rank-shaped reward does not help** (ties the control head-to-head, slightly lower on the scripted eval): training on the score margin
  already aligns with finishing order.
* **10M more steps of the plain recipe do nothing** (plain ≈ init) — the plateau is real.
* **Best checkpoint: `s5-anneal`** (+3.4 points over the bundled `s3-small`). Single seed: a
  second seed would firm up the ~3-point gap before promoting it.
