from __future__ import annotations

from pathlib import Path

import typer

from georesilience.cli.common import get_context

train_app = typer.Typer(help="Train ML models (optional: install the 'ml' extra).")


@train_app.command("gcn")
def train_gcn(
    ctx: typer.Context,
    dataset: Path = typer.Argument(
        ..., help="Dataset directory created by ingest.", dir_okay=True, file_okay=False
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        help=(
            "Run directory with node_metrics.parquet + simulation_nodes.parquet "
            "(defaults to latest)."
        ),
        dir_okay=True,
        file_okay=False,
    ),
    out_dir: Path | None = typer.Option(
        None,
        "--out-dir",
        help="Output directory for checkpoints (defaults under dataset/models).",
        dir_okay=True,
        file_okay=False,
    ),
    epochs: int = typer.Option(30, "--epochs", min=1),
    lr: float = typer.Option(1e-3, "--lr"),
    hidden_dim: int = typer.Option(64, "--hidden-dim", min=4),
    dropout: float = typer.Option(0.1, "--dropout", min=0.0, max=0.9),
    weight_decay: float = typer.Option(1e-4, "--weight-decay", min=0.0),
    val_fraction: float = typer.Option(0.2, "--val-fraction", min=0.05, max=0.5),
    seed: int = typer.Option(42, "--seed"),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda"),
) -> None:
    app_ctx = get_context(ctx.obj)

    # Lazy import to keep base install lightweight.
    from georesilience.ml.train import train_node_risk_gcn

    app_ctx.console.rule("[bold]Train[/bold]")

    ckpt_dir = train_node_risk_gcn(
        dataset_dir=dataset,
        run_dir=run_dir,
        out_dir=out_dir,
        epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        dropout=dropout,
        weight_decay=weight_decay,
        val_fraction=val_fraction,
        seed=seed,
        device=device,
    )

    app_ctx.console.print(f"[green]OK[/green] Saved checkpoints to {ckpt_dir}")
