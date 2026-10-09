# Plugins (code)

**This is where code contributions go that extend the game without changing it:** new card effect
types (mechanics), scripted agents, and anything else a `register(registry)` function can add.
Each plugin is its own **pip-installable package** with an entry point in the `wonders.extensions`
group:

```
plugins/
└── my-plugin/
    ├── pyproject.toml     # name, version, dependencies = ["wonders"], the entry point
    ├── my_plugin/
    │   ├── __init__.py    # MANIFEST + register(registry)
    │   └── cards.json     # optional: an expansion that uses the new effects
    └── tests/             # optional but recommended: run by the pull request check
```

[`example/`](example) is a complete one: a `tithe` effect type (VP per coins held), a guild that
uses it and the `taxman` agent. Start from a copy:

```
cp -r plugins/example plugins/my-plugin      # rename the package, the entry point and MANIFEST
pip install -e plugins/my-plugin
pytest plugins/my-plugin
wonders ext check plugins/my-plugin          # what the pull request check runs
wonders play -e base -e tithe -a taxman,heuristic,greedy
```

Pure data — expansions built from existing effect types, models, translations — needs no code: put
it in [`extensions/`](../extensions/README.md) instead.

## What the pull request check does

[`contributions.yml`](../.github/workflows/contributions.yml) installs the game from the base
branch, then the plugin, and runs:

* the plugin's own tests (`pytest plugins/<name>`);
* `wonders ext check plugins/<name>`: the entry points load, `MANIFEST` has id, name, author,
  licence, description and the rules version, and everything registered works — every agent plays
  base and Leaders + Cities games, every expansion goes through the same checks as a data extension,
  every model is loaded and played, every locale is checked.

Plugins run arbitrary code, so the check runs with a read-only token and no secrets, and a
maintainer reviews the code before merging. Players install plugins themselves
(`pip install plugins/<name>`); the game never downloads code.

See the [extensions guide](../docs/extensions.md#plugins-effects-agents-and-anything-else) for
the registry API.
