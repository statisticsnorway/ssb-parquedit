"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """SSB Parquedit."""


if __name__ == "__main__":
    main(prog_name="ssb-parquedit")  # pragma: no cover
