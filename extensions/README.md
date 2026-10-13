# Community extensions (data)

**This is where data contributions go:** expansions (cards and wonder boards as JSON), trained
models (`.pt` checkpoints) and the translations of an extension's own cards. One folder per
extension, named after its id:

```
extensions/
└── my-expansion/
    ├── manifest.json      # id, name, version, author, license, description, rules_version, …
    ├── cards.json         # an expansion: cards and wonders
    ├── my-net.pt          # a model, usable as neural:my-net
    ├── de.json            # its cards' names in German
    └── README.md          # optional: rules notes, how the model was trained
```

Folders hold **data only** (`.json`, `.pt`, `.md`, `.txt`, images): no code, no symbolic links,
at most 20 MB per file and 50 MB per folder. New mechanics (effect types) and scripted agents are
code: they go to [`plugins/`](../plugins/README.md). A translation of the game itself goes to
[`wonders/i18n/`](../wonders/i18n/README.md).

## Make one

```
wonders ext new my-expansion --name "My Expansion" --author "Your Name" --dir extensions
# edit extensions/my-expansion/cards.json (format: docs/extensions.md)
wonders ext check extensions/my-expansion        # what the pull request check runs
WONDERS_EXTENSIONS=extensions wonders play -e base -e my-expansion -a heuristic,board,greedy
```

A model only needs a manifest entry:

```json
"models": [{"name": "my-net", "file": "my-net.pt", "description": "PPO from the bundled network, 50M steps"}]
```

## What the pull request check does

[`contributions.yml`](../.github/workflows/contributions.yml) runs `wonders ext check` on every
changed folder, with the game from the base branch as the referee:

* **manifest** — id (lower case, the folder name), version, author, licence, description and the
  rules version it was made for; ids, expansion ids and model names unique across `extensions/`;
* **files** — data only, size limits, every file the manifest names exists;
* **cards and wonders** — every card and board builds, no name clashes, chains resolve, player
  counts within 3–7; a warning when the cards change hand sizes or leave cards undealt;
* **games** — every expansion is played with scripted agents, alone and with Leaders + Cities, at
  3 and 7 players, and each game is replayed from its record;
* **locales** — well-formed, every `{field}` kept, keys the game does not show flagged;
* **models** — loaded safely (`weights_only`), vocabulary and rules version reported, games played;
* **scorecard** — every new or changed model plays the bundled network at the same tables (seats
  shuffled, every expansion combination at 3–7 players, ~5,000 games): ranked-above rate with a 95%
  interval, first place, mean score, per combination, per player count and per score category. A
  verdict of *new state of the art* needs the interval's lower bound above 50%.

The report is posted on the pull request. After the merge,
[`leaderboard.yml`](../.github/workflows/leaderboard.yml) measures the model again and proposes a
[`LEADERBOARD.md`](../LEADERBOARD.md) update.

## Using them

Every folder here ships with the game as its **catalog** of community extensions: switch one on in
the GUI ("Community expansions") or with `wonders ext enable <id>` (`wonders ext catalog` searches them).
While you work on one, point the game at the folder itself (`WONDERS_EXTENSIONS=extensions`, several
folders separated by `:`, `;` on Windows) or copy it to `~/.wonders/extensions`: your own folder takes
precedence over the catalog's copy. See the [extensions guide](../docs/extensions.md).
