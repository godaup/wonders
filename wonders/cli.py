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
    from wonders.agents.bench import AGENTS, MAX_PLAYERS, MIN_PLAYERS, AgentSpec

    specs = []
    for entry in (a.strip() for a in agents.split(',')):
        if not entry:
            continue
        if entry == 'neural' or entry.startswith('neural:'):
            path = entry[len('neural:') :] if entry != 'neural' else _ml().pretrained_path()
            if not os.path.isfile(path):
                raise click.BadParameter(f'no checkpoint at {path!r}')
            label = f'neural:{os.path.splitext(os.path.basename(path))[0]}'
            specs.append(AgentSpec.of(label, _ml().NeuralAgent, checkpoint=os.path.abspath(path), name=label))
        elif entry in AGENTS:
            specs.append(AGENTS[entry])
        else:
            raise click.BadParameter(f"unknown agent {entry!r}; choose from {', '.join(AGENTS)} or neural:<path>")
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
    """List the built-in agents usable in 'play' and 'bench'."""
    from wonders.agents.bench import AGENTS

    for name, spec in AGENTS.items():
        doc = (spec.cls.__doc__ or '').strip().splitlines()
        click.echo(f'{name:<12} {doc[0] if doc else spec.cls.__name__}')


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
    from wonders.ml.imitation import collect, new_model, train_imitation

    data = collect(games, seed=seed, processes=processes)
    click.echo(f'recorded {len(data)} decisions from {games} games')
    model, names = new_model(hidden=hidden)
    metrics = train_imitation(model, data, epochs=epochs, seed=seed, log=click.echo)
    ml.save_checkpoint(out, model, names, source='imitation', games=games, **metrics)
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
    if init_path:
        model, card_names, _meta = ml.load_checkpoint(init_path)
    else:
        model, card_names = new_model()
    config = PPOConfig(
        opponents=tuple(names), n_envs=envs, envs_per_worker=envs_per_worker, iterations=iterations, lr=lr, seed=seed
    )
    train_ppo(model, config, log=click.echo)
    ml.save_checkpoint(out, model, card_names, source='ppo', init=init_path or '', opponents=opponents)
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


@main.command()
@click.option("--out", "-o", default="cards.html", help="Output HTML file.")
@click.option("--expansion", "-e", multiple=True, default=("base",), help="Expansions to include.")
def gallery(out, expansion):
    """Render all cards and wonder boards to a self-contained HTML gallery."""
    from wonders.cards.content import load_content
    from wonders.viz.render import gallery_html

    content = load_content(*expansion)
    title = "wonders — " + ", ".join(expansion)
    html = gallery_html(content.cards, content.wonders, title=title)
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
def gui(port, no_browser, checkpoints):
    """Play against agents in your browser: pick expansions, opponents and wonders in the menu."""
    from wonders.play.server import serve

    serve(port=port, open_browser=not no_browser, checkpoints=checkpoints)


if __name__ == "__main__":
    main()  # pragma: no cover
