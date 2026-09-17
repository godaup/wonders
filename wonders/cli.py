"""Console scripts for wonders."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("wonders")
    click.echo("=" * len("wonders"))
    click.echo("An implementation of a nice card game optimized for Deep Learning integration")


@click.command()
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


@click.command()
@click.option("--out", "-o", default="table.html", help="Output HTML file.")
@click.option("--seed", default=0, type=int, help="RNG seed for the demo game.")
@click.option("--expansion", "-e", multiple=True, default=("base",), help="Expansions to include.")
def table(out, seed, expansion):
    """Play a quick game (heuristic vs greedy vs random) and render the final table."""
    import random

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
