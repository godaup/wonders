"""Console script for wonders."""

import click


@click.command()
def main():
    """Main entrypoint."""
    click.echo("wonders")
    click.echo("=" * len("wonders"))
    click.echo("An implementation of a nice card game optimized for Deep Learning integration")


if __name__ == "__main__":
    main()  # pragma: no cover
