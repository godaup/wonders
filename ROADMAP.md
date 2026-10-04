# Roadmap

Draft, October 2026. Where `wonders` stands and what comes next, roughly in priority order.

## Where we are

- **Engine**: the base game is complete for 3–7 players: every card, both sides of every
  wonder and all wonder powers. Leaders and Cities are fully mechanized and checked
  against the official rulebooks. Armada and Babel are registered but `implemented=False`.
- **Interfaces**: an `Observation` that only shows what the seat may see, a replayable
  `GameRecord`, a fixed discrete `ActionSpace`, a PettingZoo-style `WondersEnv`,
  `SingleAgentEnv`, and sync/subprocess vector envs.
- **Agents**: random, greedy and archetype baselines, `HeuristicAgent`, the wonder-board-aware
  `BoardAgent` (about 62% wins vs heuristic + greedy), and `NeuralAgent` with the bundled
  `ppo_base_3p.pt`.
- **Learning**: imitation then PPO, a sweep runner and a benchmark with Wilson intervals.
  Universal sweep result: 256×2 is as good as or better than 512×3, so model size is not the
  bottleneck. The v2 table encoder (3,020 inputs) clones the teacher better than v1
  (validation accuracy 0.79 vs 0.74); its PPO runs are in progress.

## 1. Learning: get the most out of the v2 encoder (now)

- [ ] Finish the `ppo-C2-256x2` seeds and compare them with v1 on the same eval cells
      (vs board, vs heuristic, win rate × n).
- [ ] If v2 wins: make `table-v2` the default `encoder_kind` and retrain the bundled model as a
      universal checkpoint (all expansions and player counts) to replace `ppo_base_3p.pt`.
- [ ] Imitation overfits after about 4 epochs (accuracy peaks, then falls while the loss keeps
      dropping): add early stopping or keep the best epoch, plus weight decay / dropout sweeps.
- [ ] Self-play league: an opponent pool of past snapshots, with the bundled model and
      `BoardAgent` as fixed anchors, so PPO stops overfitting to a fixed lineup.
- [ ] Architecture to match v2's seat-relative layout: a shared per-city encoder, pooling or
      attention over the 7 seats, and card embeddings in place of one-hot blocks.

## 2. Stronger play beyond a single network pass

- [ ] Search at decision time: information-set MCTS, or determinized rollouts that sample
      hidden hands consistent with the `Observation`, guided by the policy/value net.
- [ ] Re-tune `BoardAgent` against the neural agent, so the imitation teacher and the
      baseline opponents keep up.
- [ ] Learned leader draft/recruit heads evaluated separately (the Leaders decision quality
      is not measured yet).

## 3. Training infrastructure

- [ ] Datasets on disk (memory-mapped NumPy or float16) so imitation size no longer depends on
      RAM. PR #11 brought peak memory down to one copy, about 10 GB for 12,000 v2 games.
- [ ] A durable training environment: a documented venv or container and a systemd/tmux
      launcher, instead of ad-hoc job folders.
- [ ] Run tracking: metrics.jsonl to plots, a `sweeps/report.py` summary committed per sweep,
      and checkpoint retention rules.
- [ ] A standard eval protocol (fixed seeds, cells, lineups and game counts) recorded in every
      checkpoint's metadata.

## 4. Game content

- [ ] Armada (ships, naval conflict, islands) and Babel (tower tiles, great projects), each as
      new effects and hooks, inert when the expansion is absent, plus encoder support.
- [ ] Two-player variant (the base deck currently starts at 3 players).
- [ ] Optional: the 2nd-edition rule set as its own `Ruleset`.

## 5. Library and release

- [ ] Optional Gymnasium/PettingZoo adapters (`[rl]` extra) that wrap `SingleAgentEnv` and
      `WondersEnv` for off-the-shelf RL libraries.
- [ ] A documentation site (the `doc` extra): quick start, Observation/encoder reference,
      training guide, and the `docs/encoder-v2.md` design notes.
- [ ] Move PyPI from Pre-Alpha to Beta once the engine API and checkpoint format are declared
      stable, with semver and a deprecation policy for `Observation` fields.
- [ ] Ship pretrained checkpoints as release assets if the wheel gets too large.

## 6. Playing against it

- [ ] A human seat in `wonders play` (interactive prompts showing the legal actions), against
      neural or heuristic opponents.
- [ ] A minimal web UI on top of `GameRecord` and the gallery/table renderers, including a
      replay viewer.

## Ongoing guardrails

- The determinism golden digest, the partial-observability tests (no hidden information in
  encoders) and the effect-hook design rules from CLAUDE.md.
- Performance: PPO is bound by simulation, so profile Observation building whenever new
  observation fields are added.
