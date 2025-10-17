"""Main CLI entry point for Matchbox."""

import typer

from matchbox.client.cli.eval.run import eval

app = typer.Typer(
    name="matchbox", help="Matchbox: Entity resolution and data linking framework"
)

app.command()(eval)

if __name__ == "__main__":
    app()
