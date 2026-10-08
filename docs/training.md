# Training your own agent

This guide covers the learned-agent stack: from a first network in a few minutes to the full
recipe behind the bundled model, and how to measure whether your agent is actually better.
Everything here needs the optional PyTorch extra.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,test]"
```

For an NVIDIA GPU, install the matching PyTorch wheel first, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu128`. A GPU helps the learner,
but training is mostly bound by game simulation in CPU worker processes (see
[Hardware](#hardware-and-throughput)).

## Quick start: imitation, then PPO

```
wonders train imitate -n 3000 -p 6 -o imitation.pt      # behaviour cloning of a scripted agent
wonders train ppo --init imitation.pt -n 150 -o ppo.pt  # PPO against built-in opponents
wonders train eval ppo.pt                               # vs scripted lineups, per expansion and player count
wonders play -a neural:ppo.pt,heuristic,board           # watch it play
wonders gui --checkpoint ppo.pt                         # play against it yourself
```

1. **Imitation** records games of a teacher (`HeuristicAgent` by default) in worker processes and
   clones its decisions into a policy/value network. It gives PPO a sensible starting policy and a
   trained value head (the value target is the remaining score margin).
2. **PPO** improves the network against built-in opponents (`--opponents`), with the learner seat
   rotating. `--envs` games run in parallel, `--envs-per-worker` per worker process.

The CLI covers the base game; for every expansion and player count use a sweep spec (below).

## The full recipe: sweep specs

`wonders train sweep spec.json -p 6` runs a list of training runs described in JSON. Finished runs
are skipped and interrupted ones resume, so a sweep can be stopped and restarted at any time
(`--dry-run` lists the runs, `--only name` selects some). Each run writes its checkpoint, metrics and
an evaluation report into `<out>/<name>/`.

A run is either `"kind": "imitate"` or `"kind": "ppo"`. The fields that matter most:

| field | meaning |
|---|---|
| `universal: true`, `encoder: "table-v2"` | one network for every expansion combination and 3–7 players |
| `tables` | the (expansions, player count) cells trained on, sampled per game |
| `teachers`, `games` (imitate) | scripted teachers and the number of recorded games |
| `init`, `steps` (ppo) | starting checkpoint (a file or an earlier run's name) and PPO learner steps |
| `opponent_pool` | scripted opponents; repeat a name to weight it |
| `league`, `league_size`, `snapshot_every` | self-play league: recent snapshots of the learner join the opponents |
| `tier_weights` | opponent mix: scripted / recent self / anchors (e.g. `[0.3, 0.4, 0.3]`) |
| `anchors` | fixed strong opponents (checkpoints or `run:<name>`) |
| `lr`, `lr_schedule: "linear"`, `lr_final` | learning rate and its decay (`lr_final` is a fraction of `lr`, e.g. 0.1) |
| `reward` | `dense` (default: change in score margin each step) or `dense_rank` (plus a placing bonus) |
| `n_envs`, `envs_per_worker`, `opponent_device` | parallelism; large networks as opponents on the GPU |

The bundled network was produced in two steps:

1. `sweeps/stage3.json` — imitation of the `adaptive` agent on every cell, then 40M PPO steps
   with the tiered opponent mix (`s3-small`, 256×2).
2. `sweeps/objectives.json`, run `s5-anneal` — 10M more PPO steps from `s3-small` with a linear
   learning-rate decay.

The reports next to the specs (`sweeps/*.md`) describe what was tried, what worked and what did not
(larger networks, rank rewards, exploiters).

**Note — rules changes.** The October 2026 rules fixes (trading only from the board and brown /
grey cards; buried cards and discarded leaders no longer reach the discard pile) postdate every
trained model. Train or fine-tune under the current rules before comparing with older results; see
[ROADMAP.md](../ROADMAP.md).

## Measuring progress

* `wonders train eval model.pt` — win rates against scripted lineups, per expansion combination
  and player count, with 95% confidence intervals. Good as a regression guard; between strong
  models it saturates.
* `wonders train h2h a=one.pt b=two.pt -p 6` (`bundled` names the shipped network) — the primary metric: models seated at the same
  tables (heuristic agents in the other seats), reporting how often each ranks above the other.
  Keep the default random seating: with fixed seat order the first-listed model always picks from
  a hand before the other in Ages I and III, worth about 7 points between trained models.
* `wonders train exploit e=exploiter.pt t=target.pt` — how well one model does at tables full of
  copies of another (1.0 = parity); a dedicated exploiter shows how exploitable a model is.
* `wonders train leaders model.pt` — Leaders diagnostics: agreement with a teacher per decision
  kind and ablations of the draft and recruitment.

Differences of 1–2 points need thousands of games; seed-to-seed variation of a recipe is of the
same size, so compare recipes over more than one seed.

## Hardware and throughput

* PPO is bound by simulation in the env workers (building observations for every acting seat),
  not by the network. Use about one worker per physical core plus some oversubscription — e.g.
  16 workers with 4 envs each on a 6-core / 12-thread CPU.
* The learner runs on CUDA when available; that mainly frees CPU cores for the workers.
  Two env groups are stepped in turn (`pipeline`, default 2), so simulation and inference overlap.
* Networks of 512+ units wide as opponents belong on the GPU (`opponent_device: "cuda"`): batch-1
  CPU inference of large networks is memory-bound.
* Imitation datasets live in memory: mind your RAM when raising `games`.
* Never fork after PyTorch has run: create worker pools through `wonders._mp.get_context()`.

## Training from Python

```python
from wonders.agents import HeuristicAgent, GreedyAgent
from wonders.env import SingleAgentEnv

env = SingleAgentEnv([HeuristicAgent(), GreedyAgent()], learner=0, reward='dense')
obs, info = env.reset(seed=0)
# obs['observation']: encoded features, obs['action_mask']: legal actions over the fixed action space
```

`WondersEnv` is the multi-agent version (PettingZoo-style), `SyncVectorEnv` / `SubprocVectorEnv`
batch environments with auto-reset, and `wonders.env.vocab.universal_vocab()` gives the vocabulary
and `table-v2` encoder shared by every expansion. A trained model is any checkpoint saved with
`wonders.ml.save_checkpoint(..., **vocab.meta())`; `NeuralAgent(path)` plays it anywhere an agent is
expected (CLI `-a neural:path`, the GUI, benchmarks).
