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


if __name__ == "__main__":
    main()  # pragma: no cover
