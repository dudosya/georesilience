from __future__ import annotations

from pathlib import Path

import typer

from georesilience.cli.common import get_context
from georesilience.graph.build import build_graph_from_parquet
from georesilience.graph.metrics import compute_node_metrics
from georesilience.io.parquet import read_parquet
from georesilience.sim.cascade import CascadeConfig, FailureMode, run_cascade

simulate_app = typer.Typer(help="Run resilience simulations on a dataset.")


@simulate_app.command("cascade")
def simulate_cascade(
    ctx: typer.Context,
    dataset: Path = typer.Argument(..., help="Dataset directory created by ingest.", dir_okay=True, file_okay=False),
    alpha: float = typer.Option(0.2, "--alpha", help="Capacity margin (Motter–Lai)."),
    initial_failures: int = typer.Option(1, "--initial-failures", min=1),
    mode: FailureMode = typer.Option(FailureMode.targeted, "--mode", help="Initial failure selection."),
    seed: int = typer.Option(42, "--seed", help="Random seed (for random mode)."),
    max_steps: int = typer.Option(50, "--max-steps", min=1),
) -> None:
    app_ctx = get_context(ctx.obj)

    nodes_path = dataset / "nodes.parquet"
    edges_path = dataset / "edges.parquet"

    app_ctx.console.rule("[bold]Simulate[/bold]")
    app_ctx.console.print(f"Dataset: {dataset}")

    nodes = read_parquet(nodes_path)
    edges = read_parquet(edges_path)

    graph = build_graph_from_parquet(nodes=nodes, edges=edges)
    metrics = compute_node_metrics(graph)

    cfg = CascadeConfig(
        alpha=alpha,
        initial_failures=initial_failures,
        mode=mode,
        seed=seed,
        max_steps=max_steps,
    )

    result = run_cascade(graph, node_metrics=metrics, config=cfg)

    out_dir = dataset / "runs" / f"cascade_alpha{alpha}_k{initial_failures}_{mode.value}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result.write(out_dir)
    app_ctx.console.print(f"[green]OK[/green] Wrote results to {out_dir}")
