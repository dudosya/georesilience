from __future__ import annotations

from pathlib import Path

import typer

from georesilience.cli.common import get_context
from georesilience.data.osm import fetch_power_features
from georesilience.data.pipeline import build_dataset

ingest_app = typer.Typer(help="Fetch and clean power infrastructure data.")


@ingest_app.command("city")
def ingest_city(
    ctx: typer.Context,
    place: str = typer.Argument(..., help="Place name, e.g. 'Austin, TX, USA'."),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output dataset directory (defaults under --data-dir).",
        dir_okay=True,
        file_okay=False,
    ),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable OSMnx caching."),
    validate: bool = typer.Option(
        True, "--validate/--no-validate", help="Run Pydantic validation."
    ),
) -> None:
    app_ctx = get_context(ctx.obj)
    dataset_dir = out or (app_ctx.data_dir / _slug(place))

    app_ctx.console.rule("[bold]Ingest[/bold]")
    app_ctx.console.print(f"Place: [bold]{place}[/bold]")
    app_ctx.console.print(f"Output: {dataset_dir}")

    gdf = fetch_power_features(place=place, use_cache=cache)
    build_dataset(dataset_dir=dataset_dir, place=place, features_gdf=gdf, validate=validate)

    app_ctx.console.print("[green]OK[/green] Wrote cleaned dataset")


def _slug(place: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in place).strip("-")
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe
