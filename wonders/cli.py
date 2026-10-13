"""Console scripts for wonders."""

import os
import random

import click

AGENT_HELP = (
    "Comma-separated agents, one per seat: built-in names (see 'wonders agents'), "
    "'neural' for the bundled trained network or 'neural:<checkpoint.pt>' (needs PyTorch)."
)
ML_MISSING = "this needs PyTorch: pip install 'wonders[ml]'"


def _ml():
    """Import :mod:`wonders.ml`, turning a missing PyTorch into a clean CLI error."""
    try:
        import wonders.ml as ml
    except ImportError as exc:
        raise click.ClickException(ML_MISSING) from exc
    return ml


def _lineup(agents: str):
    """Parse a lineup into agent specs (built-in names or ``neural:<path>``)."""
    from wonders.agents.bench import AGENTS, MAX_PLAYERS, MIN_PLAYERS, AgentSpec, agent_id
    from wonders.extensions import ensure_loaded

    models = ensure_loaded().models
    specs = []
    for entry in (agent_id(a.strip()) for a in agents.split(',')):  # display names work too (Pythia, ...)
        if not entry:
            continue
        if entry == 'neural' or entry.startswith('neural:'):
            path = entry[len('neural:') :] if entry != 'neural' else _ml().pretrained_path()
            if path in models:  # a model registered by an extension
                path = models[path].path
            if not os.path.isfile(path):
                raise click.BadParameter(f'no checkpoint at {path!r}')
            label = f'neural:{os.path.splitext(os.path.basename(path))[0]}'
            specs.append(AgentSpec.of(label, _ml().NeuralAgent, checkpoint=os.path.abspath(path), name=label))
        elif entry in AGENTS:
            specs.append(AGENTS[entry])
        else:
            raise click.BadParameter(
                f"unknown agent {entry!r}; choose from {', '.join(AGENTS)}, neural, neural:<path> or neural:<model>"
            )
    if not MIN_PLAYERS <= len(specs) <= MAX_PLAYERS:
        raise click.BadParameter(f'need {MIN_PLAYERS}-{MAX_PLAYERS} agents (one per seat), got {len(specs)}')
    # Seats are pooled by name, so two *different* checkpoints must not share a label.
    checkpoints: dict = {}
    for spec in specs:
        if spec.name.startswith('neural:'):
            path = dict(spec.kwargs)['checkpoint']
            if checkpoints.setdefault(spec.name, path) != path:
                raise click.BadParameter('give different checkpoints distinct file names')
    return specs


def _score_table(game) -> str:
    from wonders.agents.bench import CATEGORIES

    header = f"{'seat':<4} {'player':<14} {'wonder':<36} " + ' '.join(f'{c[:5]:>5}' for c in CATEGORIES) + '  total'
    lines = [header]
    for i, p in enumerate(game.players):
        b = game.score(i).as_dict()
        wonder = f'{p.wonder.name} ({p.wonder.side})'
        cats = ' '.join(f'{b[c]:5d}' for c in CATEGORIES)
        lines.append(f'{i:<4} {p.name:<14} {wonder:<36} {cats}  {b["total"]:5d}')
    lines.append(f'winner: {game.players[game.winner()].name}')
    return '\n'.join(lines)


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """A 7 Wonders engine for agents and deep learning."""
    if ctx.invoked_subcommand is None:
        click.echo("wonders")
        click.echo("=" * len("wonders"))
        click.echo("An implementation of a nice card game optimized for Deep Learning integration")
        click.echo("Run 'wonders --help' for the available commands.")


@main.command()
def agents():
    """List the built-in agents usable in 'play' and 'bench', with their display names and Elo ratings."""
    from wonders.agents.bench import AGENTS, agent_name
    from wonders.play.rating import agent_elo

    for name, spec in AGENTS.items():
        doc = (spec.cls.__doc__ or '').strip().splitlines()
        persona = f'{agent_name(name)} ({agent_elo(name)})'
        click.echo(f'{name:<12} {persona:<19} {doc[0] if doc else spec.cls.__name__}')


@main.command()
@click.option('--agents', '-a', default='heuristic,greedy,random', show_default=True, help=AGENT_HELP)
@click.option('--seed', default=0, type=int, show_default=True, help='RNG seed (decks, wonders, random agents).')
@click.option('--expansion', '-e', multiple=True, default=('base',), help='Expansions to include.')
@click.option('--side', type=click.Choice(['A', 'B', 'random']), default='random', show_default=True)
@click.option('--record', 'record_path', type=click.Path(dir_okay=False), help='Save the game record as JSON.')
@click.option('--quiet', '-q', is_flag=True, help='Only print the final scores.')
def play(agents, seed, expansion, side, record_path, quiet):
    """Play one game between agents and print the transcript and scores."""
    from wonders.agents.match import play_match
    from wonders.cards.content import load_content
    from wonders.engine.record import PlayerSetup

    specs = _lineup(agents)
    rng = random.Random(seed)
    content = load_content(*expansion)
    wonders = list(dict.fromkeys(w.name for w in content.wonders))
    if len(specs) > len(wonders):
        raise click.BadParameter(f'at most {len(wonders)} players with these expansions')
    setups = [
        PlayerSetup(f'{spec.name}-{i}', wonder, rng.choice('AB') if side == 'random' else side)
        for i, (spec, wonder) in enumerate(zip(specs, rng.sample(wonders, len(specs))))
    ]
    seats = [spec.build(random.Random(rng.random())) for spec in specs]
    game, record = play_match(content, setups, seats, rng=rng, expansions=tuple(expansion))
    if not quiet:
        click.echo(record.text())
        click.echo()
    click.echo(_score_table(game))
    if record_path:
        with open(record_path, 'w', encoding='utf-8') as fh:
            fh.write(record.to_json(indent=1))
        click.echo(f'record saved to {record_path}')


@main.command()
@click.argument('record_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--seat', '-s', default=0, type=int, show_default=True, help='Whose moves to rate (0 = first seat).')
@click.option('--checkpoint', type=click.Path(exists=True, dir_okay=False), help='Network to search with.')
@click.option('--depth', default=3, type=int, show_default=True, help='Turns searched per candidate.')
@click.option('--worlds', default=16, type=int, show_default=True, help='Sampled worlds per decision.')
@click.option(
    '--candidates', default=4, type=int, show_default=True, help="Network candidates besides the played move."
)
@click.option('--processes', '-p', default=4, type=int, show_default=True)
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Save the full report as JSON.')
def analyse(record_path, seat, checkpoint, depth, worlds, candidates, processes, out):
    """Rate one player's moves in a saved game with network-guided search (needs PyTorch)."""
    _ml()
    import json

    from wonders.engine.record import GameRecord
    from wonders.ml.analysis import analyse as run_analysis
    from wonders.ml.analysis import summary_text

    with open(record_path, encoding='utf-8') as fh:
        record = GameRecord.from_json(fh.read())
    if not 0 <= seat < len(record.players):
        raise click.BadParameter(f'the game has seats 0-{len(record.players) - 1}', param_hint='--seat')
    report = run_analysis(
        record,
        seat,
        checkpoint=checkpoint,
        depth=depth,
        worlds=worlds,
        candidates=candidates,
        processes=processes,
        progress=lambda done, total: click.echo(f'\ranalysed {done}/{total} moves', nl=False),
    )
    click.echo()
    click.echo(summary_text(report))
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1)
        click.echo(f'report written to {out}')


@main.command()
@click.argument('record_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Also render the final table to HTML.')
@click.option('--quiet', '-q', is_flag=True, help='Only print the final scores.')
def replay(record_path, out, quiet):
    """Re-simulate a saved game record and print its transcript and scores."""
    from wonders.cards.content import load_content
    from wonders.engine.record import GameRecord

    with open(record_path, encoding='utf-8') as fh:
        record = GameRecord.from_json(fh.read())
    game = record.replay(load_content(*record.expansions))
    if not quiet:
        click.echo(record.text())
        click.echo()
    click.echo(_score_table(game))
    if out:
        from wonders.viz.table import table_html

        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(table_html(game, title='wonders — replay'))
        click.echo(f'final table written to {out}')


@main.command()
@click.option('--agents', '-a', default='heuristic,greedy,random', show_default=True, help=AGENT_HELP)
@click.option('--games', '-n', default=200, type=int, show_default=True, help='Number of games.')
@click.option('--seed', default=0, type=int, show_default=True)
@click.option('--expansion', '-e', multiple=True, default=('base',), help='Expansions to include.')
@click.option('--side', type=click.Choice(['A', 'B', 'random']), default='random', show_default=True)
@click.option('--processes', '-p', default=1, type=int, show_default=True, help='Worker processes.')
def bench(agents, games, seed, expansion, side, processes):
    """Benchmark agents over many games (seats rotate; win rates with 95% CIs)."""
    from wonders.agents.bench import run_bench

    result = run_bench(
        _lineup(agents), n_games=games, seed=seed, expansions=tuple(expansion), sides=side, processes=processes
    )
    click.echo(result.table())


@main.group()
def ext():
    """Extensions: custom expansions, card effects, agents, models and translations."""


@ext.command(name='list')
def ext_list():
    """Show the installed extensions (folders and plugins) and what each provides."""
    from wonders.extensions import ensure_loaded, extensions_homes

    registry = ensure_loaded()
    click.echo(f'extensions home: {os.pathsep.join(map(str, extensions_homes()))}')
    if not registry.extensions:
        click.echo('no extensions installed (browse the community ones: wonders ext catalog)')
        return
    for ext_id, info in sorted(registry.extensions.items()):
        m = info.manifest
        click.echo(f'{ext_id} {m.version} [{info.source}] {m.name} — hash {info.digest or "-"}')
        if m.rules_version is not None:
            from wonders.ruleset import RULES_VERSION

            note = '' if m.rules_version == RULES_VERSION else f' (engine implements {RULES_VERSION})'
            click.echo(f'  made for rules version {m.rules_version}{note}')
        click.echo('  provides: ' + (', '.join(info.provides) or 'nothing'))


@ext.command(name='catalog')
@click.argument('query', nargs=-1)
def ext_catalog(query):
    """Search the community extensions shipped with the game (every word must match; none: list all)."""
    from wonders.extensions import catalog, catalog_homes

    entries = catalog(' '.join(query))
    if not entries:
        where = os.pathsep.join(map(str, catalog_homes())) or 'no catalog folder'
        raise click.ClickException(f'no catalog extension matches {" ".join(query)!r} ({where})')
    for entry in entries:
        info = entry.to_dict()
        state = 'on' if entry.active else 'off'
        if entry.loaded and entry.loaded != 'catalog':
            state = f'installed as a {entry.loaded}'
        click.echo(
            f'{info["id"]} {info["version"]} [{state}] {info["name"]}'
            + (f' by {info["author"]}' if info['author'] else '')
        )
        if info['description']:
            click.echo(f'  {info["description"]}')
        parts = [f'{info["cards"]} cards'] if info['cards'] else []
        parts += [f'boards: {", ".join(info["wonders"])}'] if info['wonders'] else []
        parts += [f'colours: {", ".join(info["colors"])}'] if info['colors'] else []
        if parts:
            click.echo('  ' + '; '.join(parts))
    click.echo('switch one on with: wonders ext enable <id>')


@ext.command(name='enable')
@click.argument('extension_id')
def ext_enable(extension_id):
    """Switch a catalog extension on: its expansions join the menu and the CLI from now on."""
    from wonders.extensions import activate

    try:
        entry = activate(extension_id, True)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    names = ', '.join(e.get('name', e['id']) for e in entry.manifest.expansions) or entry.manifest.name
    click.echo(f'{entry.manifest.id} is on: {names} can be chosen (e.g. wonders play -e base -e {entry.manifest.id})')


@ext.command(name='disable')
@click.argument('extension_id')
def ext_disable(extension_id):
    """Switch a catalog extension off (saved games that used it still replay)."""
    from wonders.extensions import activate

    try:
        activate(extension_id, False)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    click.echo(f'{extension_id} is off')


@ext.command(name='validate')
@click.argument('folder', type=click.Path(exists=True, file_okay=False))
def ext_validate(folder):
    """Check an extension folder: manifest, files, cards, wonders, names and chains."""
    from wonders.extensions import validate

    problems = validate(folder)
    for problem in problems:
        click.echo(f'✗ {problem}')
    if problems:
        raise click.ClickException(f'{len(problems)} problem(s) in {folder}')
    click.echo(f'✓ {folder} is a valid extension')


@ext.command(name='new')
@click.argument('extension_id')
@click.option('--name', default='', help='Display name (defaults to the id).')
@click.option('--author', default='', help='Your name (contributions need one).')
@click.option(
    '--dir', 'parent', type=click.Path(file_okay=False), help='Where to create it (default: the extensions home).'
)
def ext_new(extension_id, name, author, parent):
    """Create a minimal data extension (manifest + one example card) to start from."""
    from wonders.extensions import extensions_home, scaffold

    folder = scaffold(os.path.join(parent or extensions_home(), extension_id), extension_id, name, author)
    click.echo(f'created {folder}: edit cards.json, then run  wonders ext check {folder}')


@ext.command(name='check')
@click.argument('targets', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--games', '-n', default=4, type=int, show_default=True, help='Games per smoke test.')
@click.option('--report', type=click.Path(dir_okay=False), help='Also write the Markdown report here.')
def ext_check(targets, games, report):
    """Check contributions: extension folders, installed plugin packages and locale files.

    Stricter than ``validate``: manifest metadata, file types and sizes, clashes with other extensions,
    deck sizes, games played and replayed with every expansion, agent and model, and translations.
    """
    from wonders.contrib import check

    reports = []
    for target in targets:
        try:
            reports.append(check(target, games))
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc
    text = '\n'.join(r.markdown() for r in reports)
    click.echo(text)
    if report:
        with open(report, 'w', encoding='utf-8') as fh:
            fh.write(text)
    failed = [r.target for r in reports if not r.passed]
    if failed:
        raise click.ClickException(f'checks failed for {", ".join(failed)}')


@ext.command(name='changed')
@click.argument('paths', nargs=-1)
@click.option('--root', default='.', type=click.Path(file_okay=False), show_default=True)
@click.option('--github-output', type=click.Path(dir_okay=False), help='Append key=JSON lines for GitHub Actions.')
def ext_changed(paths, root, github_output):
    """Which contributions the changed PATHS touch (or one path per line on stdin)."""
    import json
    import sys

    from wonders.contrib import changed

    result = changed(paths or sys.stdin.read().splitlines(), root)
    click.echo(json.dumps(result, indent=1))
    if github_output:
        with open(github_output, 'a', encoding='utf-8') as fh:
            for key, value in result.items():
                fh.write(f'{key}={json.dumps(value)}\n')
    if result['invalid']:
        allowed = 'names must be lower-case letters, digits, ".", "_" or "-"'
        raise click.ClickException(f"{allowed}: {', '.join(result['invalid'])}")


@ext.command(name='scorecard')
@click.argument('source', type=click.Path(exists=True))
@click.option('--model', help='Which model of the extension (needed when it has several).')
@click.option('--cells', default='all', show_default=True, help="'all' or e.g. base/3p,leaders+cities/5p")
@click.option('--games', '-n', default=200, type=int, show_default=True, help='Games per cell.')
@click.option('--seed', default=4242, type=int, show_default=True)
@click.option('--processes', '-p', default=4, type=int, show_default=True)
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Save the scorecard JSON (for summary).')
def ext_scorecard(source, model, cells, games, seed, processes, out):
    """Play a contributed model (extension folder or .pt) against the bundled network (needs PyTorch)."""
    import json

    _ml()
    from wonders.contrib import scorecard, scorecard_markdown

    try:
        card = scorecard(source, model, cells=cells, games=games, seed=seed, processes=processes)
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(card, fh, indent=1)
    click.echo(scorecard_markdown(card))


@ext.command(name='summary')
@click.argument('directory', type=click.Path(exists=True, file_okay=False))
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Write the Markdown here as well.')
def ext_summary(directory, out):
    """One report from a folder of check reports (*.md) and scorecards (*.json; shards are merged)."""
    from wonders.contrib import summary

    text = summary(directory)
    click.echo(text)
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text)


@ext.command(name='leaderboard')
@click.argument('scorecards', nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--data', default='leaderboard.json', show_default=True, type=click.Path(dir_okay=False))
@click.option('--markdown', default='LEADERBOARD.md', show_default=True, type=click.Path(dir_okay=False))
def ext_leaderboard(scorecards, data, markdown):
    """Add scorecards (shards of one model are merged) to the leaderboard and re-render it."""
    import json

    from wonders.contrib import merge_scorecards, update_leaderboard

    cards = []
    for path in scorecards:
        with open(path, encoding='utf-8') as fh:
            cards.append(json.load(fh))
    for card in merge_scorecards(cards):
        update_leaderboard(card, data, markdown)
        click.echo(f"leaderboard: {card.get('extension', '')}/{card['model']}")


@main.group()
def train():
    """Train a neural agent (needs PyTorch: pip install 'wonders[ml]')."""


@train.command()
@click.option('--games', '-n', default=3000, type=int, show_default=True, help='Teacher games to record.')
@click.option('--epochs', default=12, type=int, show_default=True)
@click.option('--hidden', default=256, type=int, show_default=True, help='Hidden layer width.')
@click.option('--seed', default=0, type=int, show_default=True)
@click.option('--processes', '-p', default=1, type=int, show_default=True, help='Processes for game collection.')
@click.option('--out', '-o', default='imitation.pt', show_default=True, type=click.Path(dir_okay=False))
def imitate(games, epochs, hidden, seed, processes, out):
    """Clone HeuristicAgent's play into a network (behaviour cloning)."""
    ml = _ml()
    from wonders.env.vocab import vocab_for
    from wonders.ml.imitation import collect, new_model, train_imitation

    vocab = vocab_for(('base',))
    data = collect(games, seed=seed, processes=processes, vocab=vocab)
    click.echo(f'recorded {len(data)} decisions from {games} games')
    model, names = new_model(hidden=hidden, vocab=vocab)
    metrics = train_imitation(model, data, epochs=epochs, seed=seed, log=click.echo)
    ml.save_checkpoint(out, model, names, source='imitation', games=games, **vocab.meta(), **metrics)
    click.echo(f'saved {out}')


@train.command()
@click.option('--init', 'init_path', type=click.Path(exists=True, dir_okay=False), help='Start from a checkpoint.')
@click.option('--iterations', '-n', default=100, type=int, show_default=True)
@click.option('--opponents', default='heuristic,greedy', show_default=True, help='Built-in opponent names.')
@click.option('--envs', default=24, type=int, show_default=True, help='Parallel games.')
@click.option('--envs-per-worker', default=4, type=int, show_default=True, help='0 = no worker processes.')
@click.option('--lr', default=1e-4, type=float, show_default=True)
@click.option('--seed', default=0, type=int, show_default=True)
@click.option('--out', '-o', default='ppo.pt', show_default=True, type=click.Path(dir_okay=False))
def ppo(init_path, iterations, opponents, envs, envs_per_worker, lr, seed, out):
    """Improve a network with PPO against built-in opponents."""
    ml = _ml()
    from wonders.agents.bench import AGENTS, MAX_PLAYERS, MIN_PLAYERS
    from wonders.ml.imitation import new_model
    from wonders.ml.ppo import PPOConfig, train_ppo

    names = [n.strip() for n in opponents.split(',') if n.strip()]
    unknown = [n for n in names if n not in AGENTS]
    if unknown or not MIN_PLAYERS <= len(names) + 1 <= MAX_PLAYERS:
        raise click.BadParameter(f'need {MIN_PLAYERS - 1}-{MAX_PLAYERS - 1} built-in opponents; unknown: {unknown}')
    from wonders.ml.ppo import config_vocab

    encoder = None
    if init_path:  # keep the encoder the starting checkpoint was built for (older ones: v1)
        model, card_names, meta = ml.load_checkpoint(init_path)
        encoder = meta.get('encoder', 'card-v1')
    config = PPOConfig(
        opponents=tuple(names), n_envs=envs, envs_per_worker=envs_per_worker, iterations=iterations, lr=lr, seed=seed
    )
    if encoder is not None:
        config.encoder = encoder
    vocab = config_vocab(config)
    if not init_path:
        model, card_names = new_model(vocab=vocab)
    train_ppo(model, config, log=click.echo)
    ml.save_checkpoint(out, model, card_names, source='ppo', init=init_path or '', opponents=opponents, **vocab.meta())
    click.echo(f'saved {out}')


@train.command()
@click.argument('checkpoint', type=click.Path(exists=True, dir_okay=False))
@click.option('--games', '-n', default=2000, type=int, show_default=True, help='Games with one searching seat.')
@click.option('--cells', default='all', show_default=True, help="'all' or e.g. base/3p,leaders+cities/5p")
@click.option('--depth', default=3, type=int, show_default=True, help='Turns searched before the value head.')
@click.option('--worlds', default=16, type=int, show_default=True, help='Sampled worlds per decision.')
@click.option('--candidates', default=4, type=int, show_default=True, help='Moves searched per decision.')
@click.option(
    '--temperature',
    default=1.0,
    type=float,
    show_default=True,
    help='VP; soft-target temperature (also re-applied to loaded --data).',
)
@click.option('--opponents', default='neural', show_default=True, help="Other seats: 'neural' or built-in names.")
@click.option('--epochs', default=3, type=int, show_default=True)
@click.option('--lr', default=1e-4, type=float, show_default=True)
@click.option('--seed', default=0, type=int, show_default=True)
@click.option('--processes', '-p', default=6, type=int, show_default=True)
@click.option(
    '--data', 'data_path', type=click.Path(dir_okay=False), help='Save the search data here (reused if present).'
)
@click.option('--out', '-o', default='expert.pt', show_default=True, type=click.Path(dir_okay=False))
def expert(
    checkpoint,
    games,
    cells,
    depth,
    worlds,
    candidates,
    temperature,
    opponents,
    epochs,
    lr,
    seed,
    processes,
    data_path,
    out,
):
    """Expert iteration: record network-guided search with CHECKPOINT, then distil it back into the network."""
    ml = _ml()
    from wonders.ml.evaluate import parse_cells
    from wonders.ml.expert import SearchDataset, collect_search, distill
    from wonders.ruleset import RULES_VERSION

    if data_path and os.path.isfile(data_path):
        data = SearchDataset.load(data_path)
        click.echo(f'loaded {len(data)} decisions from {data_path}')
    else:
        names = [n.strip() for n in opponents.split(',') if n.strip()]
        data = collect_search(
            checkpoint,
            games,
            parse_cells(cells),
            depth=depth,
            worlds=worlds,
            candidates=candidates,
            temperature=temperature,
            opponents=names,
            seed=seed,
            processes=processes,
            log=click.echo,
        )
        click.echo(f'recorded {len(data)} decisions from {games} games')
        if data_path:
            data.save(data_path)
    model, card_names, meta = ml.load_checkpoint(checkpoint)
    metrics = distill(model, data, epochs=epochs, lr=lr, seed=seed, log=click.echo, temperature=temperature)
    meta.update(source='expert', init=os.path.abspath(checkpoint), rules_version=RULES_VERSION, **metrics)
    ml.save_checkpoint(out, model, card_names, **meta)
    click.echo(f'saved {out}')


@train.command()
@click.argument('spec', type=click.Path(exists=True, dir_okay=False))
@click.option('--only', multiple=True, help='Only runs whose name starts with this (repeatable).')
@click.option('--processes', '-p', default=6, type=int, show_default=True, help='Worker processes.')
@click.option('--dry-run', is_flag=True, help='List the runs without executing them.')
def sweep(spec, only, processes, dry_run):
    """Run a training sweep spec (JSON); finished runs are skipped, interrupted ones resume."""
    _ml()
    from wonders.ml.sweep import run_sweep

    run_sweep(spec, only=only, processes=processes, dry_run=dry_run, log=click.echo)


@train.command(name='eval')
@click.argument('checkpoint', type=click.Path(exists=True, dir_okay=False))
@click.option('--cells', default='all', show_default=True, help="'all' or e.g. base/3p,leaders+cities/5p")
@click.option('--lineups', default='heuristic,board', show_default=True, help='Reference lineups.')
@click.option('--games', '-n', default=600, type=int, show_default=True)
@click.option('--seed', default=4242, type=int, show_default=True)
@click.option('--processes', '-p', default=6, type=int, show_default=True)
def evaluate_command(checkpoint, cells, lineups, games, seed, processes):
    """Evaluate a checkpoint per expansion combination and player count (win rate × players)."""
    _ml()
    from wonders.ml.evaluate import evaluate, parse_cells, summary_table

    results = evaluate(
        checkpoint,
        cells=parse_cells(cells),
        lineups=tuple(x.strip() for x in lineups.split(',')),
        games=games,
        seed=seed,
        processes=processes,
    )
    click.echo(summary_table(results, checkpoint))


def _model_arg(entry: str):
    """``label=path`` or a bare path (labelled by its run directory when the file is ``final.pt``).

    ``bundled`` (as the path) stands for the network shipped with the package.
    """
    label, sep, path = entry.partition('=')
    if (path if sep else entry) == 'bundled':
        from wonders.ml import pretrained_path

        return (label if sep else 'bundled'), pretrained_path()
    if not sep:
        path = entry
        stem = os.path.splitext(os.path.basename(path))[0]
        label = os.path.basename(os.path.dirname(os.path.abspath(path))) if stem in ('final', 'state') else stem
    if not os.path.isfile(path):
        raise click.BadParameter(f'no checkpoint at {path!r}')
    return label, path


@train.command(name='h2h')
@click.argument('models', nargs=-1, required=True)
@click.option('--cells', default='all', show_default=True, help="'all' or e.g. base/3p,leaders+cities/5p")
@click.option('--filler', default='heuristic', show_default=True, help='Agent filling the remaining seats.')
@click.option(
    '--seating',
    type=click.Choice(['shuffle', 'rotate']),
    default='shuffle',
    show_default=True,
    help='shuffle: random seats per game; rotate: fixed order (biased towards the first model).',
)
@click.option('--games', '-n', default=400, type=int, show_default=True, help='Games per cell.')
@click.option('--seed', default=4242, type=int, show_default=True)
@click.option('--processes', '-p', default=6, type=int, show_default=True)
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Also save the full results as JSON.')
def head_to_head_command(models, cells, filler, seating, games, seed, processes, out):
    """Play checkpoints against each other at the same tables (MODELS: label=path, path or bundled)."""
    _ml()
    import json

    from wonders.agents.bench import AGENTS
    from wonders.ml.evaluate import head_to_head, head_to_head_table, parse_cells

    if filler not in AGENTS:
        raise click.BadParameter(f'unknown agent {filler!r}')
    parsed = [_model_arg(m) for m in models]
    if len(parsed) < 2:
        raise click.BadParameter('give at least two checkpoints')
    result = head_to_head(
        parsed,
        cells=parse_cells(cells),
        games=games,
        seed=seed,
        filler=filler,
        processes=processes,
        seating=seating,
    )
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=1)
    click.echo(head_to_head_table(result))


@train.command(name='exploit')
@click.argument('exploiter')
@click.argument('target')
@click.option('--cells', default='all', show_default=True, help="'all' or e.g. base/3p,leaders+cities/5p")
@click.option('--games', '-n', default=400, type=int, show_default=True, help='Games per cell.')
@click.option('--seed', default=4242, type=int, show_default=True)
@click.option('--processes', '-p', default=6, type=int, show_default=True)
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Also save the full results as JSON.')
def exploit_command(exploiter, target, cells, games, seed, processes, out):
    """Measure how well EXPLOITER does at tables full of TARGET copies (win rate × players; 1.0 = parity)."""
    _ml()
    import json

    from wonders.ml.evaluate import exploitability, parse_cells

    result = exploitability(
        _model_arg(exploiter), _model_arg(target), cells=parse_cells(cells), games=games, seed=seed, processes=processes
    )
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(result, fh, indent=1)
    click.echo(
        f"{result['exploiter']} vs {result['target']} field: "
        f"win × n {result['win_rel']:.3f} ± {result['win_rel_se']:.3f}"
    )
    for c in result['cells']:
        lo, hi = c['win_rel_ci']
        click.echo(f"  {c['cell']:18s} {c['win_rel']:.2f} ({lo:.2f}–{hi:.2f})  mean rank {c['mean_rank']:.2f}")


@train.command(name='leaders')
@click.argument('checkpoint')
@click.option('--teacher', default='board', show_default=True, help='Scripted agent to compare against.')
@click.option('--games', '-n', default=200, type=int, show_default=True, help='Ablation games per cell.')
@click.option('--agreement-games', default=50, type=int, show_default=True, help='Teacher-only games per cell.')
@click.option('--seed', default=4242, type=int, show_default=True)
@click.option('--processes', '-p', default=6, type=int, show_default=True)
@click.option('--out', '-o', type=click.Path(dir_okay=False), help='Also save the full report as JSON.')
def leaders_command(checkpoint, teacher, games, agreement_games, seed, processes, out):
    """Diagnose Leaders play: teacher agreement per decision kind and leader-decision ablations."""
    _ml()
    import json

    from wonders.ml.diagnostics import leaders_report, leaders_report_table

    label, path = _model_arg(checkpoint)
    report = leaders_report(
        path, label, teacher=teacher, games=games, agreement_games=agreement_games, seed=seed, processes=processes
    )
    if out:
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=1)
    click.echo(leaders_report_table(report, f'Leaders diagnostics: {label}'))


@main.command()
@click.option("--out", "-o", default="cards.html", help="Output HTML file.")
@click.option("--expansion", "-e", multiple=True, default=("base",), help="Expansions to include.")
@click.option("--lang", default="en", show_default=True, help="Language of the card texts (e.g. de).")
def gallery(out, expansion, lang):
    """Render all cards and wonder boards to a self-contained HTML gallery."""
    from wonders.cards.content import load_content
    from wonders.i18n import Translator
    from wonders.viz.render import gallery_html

    content = load_content(*expansion)
    title = "wonders — " + ", ".join(expansion)
    html = gallery_html(content.cards, content.wonders, title=title, t=Translator(lang))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    click.echo(f"Wrote {len(content.cards)} cards and {len(content.wonders)} wonders to {out}")


@main.command()
@click.option("--out", "-o", default="table.html", help="Output HTML file.")
@click.option("--seed", default=0, type=int, help="RNG seed for the demo game.")
@click.option("--expansion", "-e", multiple=True, default=("base",), help="Expansions to include.")
def table(out, seed, expansion):
    """Play a quick game (heuristic vs greedy vs random) and render the final table."""
    from wonders.agents import GreedyAgent, HeuristicAgent, RandomAgent
    from wonders.agents.match import play_match
    from wonders.cards.content import load_content
    from wonders.engine.record import PlayerSetup
    from wonders.viz.table import table_html

    content = load_content(*expansion)
    setups = [
        PlayerSetup("Heuristic", "The Pyramids of Giza"),
        PlayerSetup("Greedy", "The Colossus of Rhodes"),
        PlayerSetup("Random", "The Temple of Artemis in Ephesus"),
    ]
    agents = [HeuristicAgent(), GreedyAgent(), RandomAgent(random.Random(seed))]
    game, _ = play_match(content, setups, agents, rng=random.Random(seed), expansions=tuple(expansion))
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(table_html(game, title="wonders — final table"))
    click.echo(f"Wrote final table ({game.players[game.winner()].name} won) to {out}")


@main.command()
@click.option('--port', default=8765, type=int, show_default=True, help='Local port (0 = any free port).')
@click.option('--no-browser', is_flag=True, help='Do not open a browser window.')
@click.option(
    '--checkpoint',
    'checkpoints',
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help='Offer this trained network as an opponent too (repeatable; needs PyTorch).',
)
@click.option('--no-save', is_flag=True, help='Do not save games (by default every game goes to $WONDERS_HOME/games).')
def gui(port, no_browser, checkpoints, no_save):
    """Play against agents in your browser: pick expansions, opponents and wonders in the menu.

    Every game is saved as you play, so you can resume it later; the menu shows your past games, your
    Elo rating against the agents and your achievements.
    """
    from wonders.play.server import serve

    serve(port=port, open_browser=not no_browser, checkpoints=checkpoints, save=not no_save)


if __name__ == "__main__":
    main()  # pragma: no cover
