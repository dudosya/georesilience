from __future__ import annotations

from pathlib import Path

import typer

from georesilience.cli.common import get_context

predict_app = typer.Typer(help="Run ML prediction (optional: install the 'ml' extra).")


@predict_app.command("gcn")
def predict_gcn(
    ctx: typer.Context,
    dataset: Path = typer.Argument(
        ..., help="Dataset directory created by ingest.", dir_okay=True, file_okay=False
    ),
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        exists=True,
        dir_okay=False,
        file_okay=True,
        help="Lightning .ckpt file.",
    ),
    run_dir: Path | None = typer.Option(
        None,
        "--run-dir",
        help="Run directory used to build features (defaults to latest).",
        dir_okay=True,
        file_okay=False,
    ),
    out: Path | None = typer.Option(
        None,
        "--out",
        help="Output Parquet path (defaults under dataset/predictions).",
        dir_okay=False,
        file_okay=True,
    ),
    top_k: int = typer.Option(20, "--top-k", min=1),
    device: str = typer.Option("auto", "--device", help="auto|cpu|cuda"),
) -> None:
    app_ctx = get_context(ctx.obj)

    from georesilience.ml.inference import predict_node_risk

    app_ctx.console.rule("[bold]Predict[/bold]")

    pred_path = predict_node_risk(
        dataset_dir=dataset,
        run_dir=run_dir,
        checkpoint=checkpoint,
        out_path=out,
        top_k=top_k,
        device=device,
        console=app_ctx.console,
    )

    app_ctx.console.print(f"[green]OK[/green] Wrote predictions to {pred_path}")
