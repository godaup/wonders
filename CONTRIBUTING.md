# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/godaup/wonders/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

wonders could always use more documentation, whether as part of the
official wonders docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at https://github.com/godaup/wonders/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

### Contribute content: expansions, models, translations, plugins

Content does not need changes to the engine. Each kind has its own place, and a pull request that
touches one gets an automatic check with a report posted on the pull request
([`contributions.yml`](.github/workflows/contributions.yml)):

| you contribute | it goes to | checked by |
|---|---|---|
| an expansion (cards, wonder boards) built from existing effect types | [`extensions/<id>/`](extensions/README.md) (data) | `wonders ext check`: manifest, cards, decks, games played and replayed |
| a trained model | [`extensions/<id>/`](extensions/README.md) (data) | the same, plus a **scorecard** against the bundled network |
| the names of an extension's cards in another language | its [`extensions/<id>/`](extensions/README.md) folder | `wonders ext check`: placeholders, keys |
| a translation of the game | [`wonders/i18n/<lang>.json`](wonders/i18n/README.md) | `wonders ext check`: placeholders, keys, coverage |
| a new effect type (mechanic), a scripted agent | [`plugins/<name>/`](plugins/README.md) (a pip package) | its own tests + `wonders ext check` |

Run the same check locally before opening the pull request, e.g.
`wonders ext check extensions/my-expansion`. For data contributions the referee is the base
branch: the game and the checks come from it, only your files from the pull request. A model's
scorecard reports how often it finishes above the bundled network (95% interval) overall, per
expansion combination, per player count and per score category; merged models are measured again
for the [leaderboard](LEADERBOARD.md). Format details: [extensions guide](docs/extensions.md).

## Get Started!

Ready to contribute? Here's how to set up `wonders` for local development.

1. Fork the `wonders` repo on GitHub.
2. Clone your fork locally

    ```
    $ git clone git@github.com:your_name_here/wonders.git
    ```

3. Create and activate a virtual environment:

    ```
    $ python -m venv .venv
    $ source .venv/bin/activate   # Windows: .venv\Scripts\activate
    ```

4. Install the package with its development dependencies (editable install):

    ```
    $ pip install -e ".[test,dev]"
    ```

5. Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

6. When you're done making changes, check that your changes pass the
   tests, including testing other Python versions, with tox:

    ```
    $ tox
    ```

7. Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

8. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.11, 3.12, 3.13 and 3.14. Check
   https://github.com/godaup/wonders/actions
   and make sure that the tests pass for all supported Python versions.

## Tips

```
$ pytest tests/test_wonders.py
```

To run a subset of tests.


## Deploying

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in CHANGELOG.md).
Then run:

```
$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags
```

GitHub Actions will then deploy to PyPI if tests pass.
