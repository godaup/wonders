# Translations of the game

**This is where a translation of the game itself goes:** one flat JSON file per language, named
after its code (`fr.json`, `pt-BR.json`), mapping the English text to the translation. It ships with
the package, and the GUI offers it in the language picker straight away.

`de.json` is the reference: its keys are every text the game shows (GUI, cards, wonder boards,
achievements, the post-game analysis). Start from a copy:

```
cp wonders/i18n/de.json wonders/i18n/fr.json   # then translate the values, set "_language"
wonders ext check wonders/i18n/fr.json         # what the pull request check runs
wonders gui                                    # pick the language in the header, or ?lang=fr
```

* `_language` is the name in the picker, `_decimal` the decimal separator, `_ordinal` how ranks are
  written (e.g. `"{n}."`).
* Keep every `{field}` of the English text; `"text|context"` keys separate words with two meanings.
* A partial translation is fine: missing texts stay English, and the check reports the coverage.

The pull request check ([`contributions.yml`](../../.github/workflows/contributions.yml)) verifies
the file name, `_language`, duplicates, empty values and `{fields}`, flags keys the game does not
show, and reports how much of the game is covered. The names of an **extension's** cards are
translated in the extension's own folder ([`extensions/`](../../extensions/README.md)). Format
details: [extensions guide](../../docs/extensions.md#translations).
