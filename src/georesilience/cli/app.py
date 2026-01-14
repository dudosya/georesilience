from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from georesilience.cli.ingest import ingest_app
from georesilience.cli.simulate import simulate_app

app = typer.Typer(
    name="georesilience",
    help="GeoResilience: power-grid digital twin for resilience simulation and risk prediction.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def _main(
    ctx: typer.Context,
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        help="Base directory for datasets and outputs.",
        dir_okay=True,
        file_okay=False,
    ),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


app.add_typer(ingest_app, name="ingest")
app.add_typer(simulate_app, name="simulate")


def main() -> None:
    app()
