from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import typer

from georesilience.cli.common import get_context
from georesilience.graph.build import build_graph_from_parquet
from georesilience.graph.metrics import compute_node_metrics, edge_metrics_frame, node_metrics_frame
from georesilience.io.parquet import read_parquet
from georesilience.sim.cascade import CascadeConfig, FailureMode, run_cascade
from georesilience.util.seed import seed_everything

simulate_app = typer.Typer(help="Run resilience simulations on a dataset.")


@simulate_app.command("cascade")
def simulate_cascade(
    ctx: typer.Context,
    dataset: Path = typer.Argument(
        ..., help="Dataset directory created by ingest.", dir_okay=True, file_okay=False
    ),
    alpha: float = typer.Option(0.2, "--alpha", help="Capacity margin (Motter–Lai)."),
    initial_failures: int = typer.Option(1, "--initial-failures", min=1),
    mode: FailureMode = typer.Option(
        FailureMode.targeted, "--mode", help="Initial failure selection."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed (for random mode)."),
    max_steps: int = typer.Option(50, "--max-steps", min=1),
    advanced_metrics: bool = typer.Option(
        False,
        "--advanced-metrics",
        help="Compute slower metrics (e.g., eigenvector centrality) for features.",
    ),
) -> None:
    app_ctx = get_context(ctx.obj)

    nodes_path = dataset / "nodes.parquet"
    edges_path = dataset / "edges.parquet"

    app_ctx.console.rule("[bold]Simulate[/bold]")
    app_ctx.console.print(f"Dataset: {dataset}")

    seed_everything(seed)

    nodes = read_parquet(nodes_path)
    edges = read_parquet(edges_path)

    graph = build_graph_from_parquet(nodes=nodes, edges=edges)
    metrics = compute_node_metrics(graph, advanced=advanced_metrics)

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

    # Export features for downstream ML.
    node_metrics_frame(metrics).write_parquet(out_dir / "node_metrics.parquet")
    edge_metrics_frame(graph, advanced=advanced_metrics).write_parquet(
        out_dir / "edge_metrics.parquet"
    )

    result.write(out_dir)

    # Stable, human-readable run metadata.
    failed_count = int(result.nodes.filter(result.nodes["failed"]).height)
    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": str(dataset),
        "node_count": int(graph.number_of_nodes()),
        "edge_count": int(graph.number_of_edges()),
        "failed_nodes": failed_count,
        "failed_fraction": float(
            (failed_count / result.nodes.height) if result.nodes.height else 0.0
        ),
        "steps": int(result.steps.height),
        "advanced_metrics": bool(advanced_metrics),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "tool": "georesilience",
        "command": "simulate cascade",
        "config": cfg.__dict__,
        "summary": summary,
        "artifacts": {
            "simulation_nodes": "simulation_nodes.parquet",
            "simulation_steps": "simulation_steps.parquet",
            "node_metrics": "node_metrics.parquet",
            "edge_metrics": "edge_metrics.parquet",
            "config": "config.json",
            "summary": "summary.json",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    app_ctx.console.print(f"[green]OK[/green] Wrote results to {out_dir}")
