"""Console scripts for wonders."""

import random

import click

AGENT_HELP = "Comma-separated agent names, one per seat (see 'wonders agents')."


def _lineup(agents: str):
    from wonders.agents.bench import AGENTS, MAX_PLAYERS, MIN_PLAYERS

    names = [a.strip() for a in agents.split(',') if a.strip()]
    unknown = [n for n in names if n not in AGENTS]
    if unknown:
        raise click.BadParameter(f"unknown agent(s) {', '.join(unknown)}; choose from {', '.join(AGENTS)}")
    if not MIN_PLAYERS <= len(names) <= MAX_PLAYERS:
        raise click.BadParameter(f'need {MIN_PLAYERS}-{MAX_PLAYERS} agents (one per seat), got {len(names)}')
    return names


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
    from wonders.agents.bench import AGENTS
    from wonders.agents.match import play_match
    from wonders.cards.content import load_content
    from wonders.engine.record import PlayerSetup

    names = _lineup(agents)
    rng = random.Random(seed)
    content = load_content(*expansion)
    wonders = list(dict.fromkeys(w.name for w in content.wonders))
    if len(names) > len(wonders):
        raise click.BadParameter(f'at most {len(wonders)} players with these expansions')
    setups = [
        PlayerSetup(f'{name}-{i}', wonder, rng.choice('AB') if side == 'random' else side)
        for i, (name, wonder) in enumerate(zip(names, rng.sample(wonders, len(names))))
    ]
    seats = [AGENTS[name].build(random.Random(rng.random())) for name in names]
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


if __name__ == "__main__":
    main()  # pragma: no cover
