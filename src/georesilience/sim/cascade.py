from __future__ import annotations

import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import networkx as nx
import polars as pl

from georesilience.graph.metrics import NodeMetrics
from georesilience.io.parquet import write_parquet


class FailureMode(str, Enum):
    targeted = "targeted"
    random = "random"


@dataclass(frozen=True)
class CascadeConfig:
    alpha: float
    initial_failures: int
    mode: FailureMode
    seed: int
    max_steps: int


@dataclass(frozen=True)
class CascadeResult:
    nodes: pl.DataFrame
    steps: pl.DataFrame
    config: CascadeConfig

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_parquet(self.nodes, out_dir / "simulation_nodes.parquet")
        write_parquet(self.steps, out_dir / "simulation_steps.parquet")
        (out_dir / "config.json").write_text(
            json.dumps(self.config.__dict__, indent=2), encoding="utf-8"
        )


def run_cascade(
    graph: nx.Graph, *, node_metrics: NodeMetrics, config: CascadeConfig
) -> CascadeResult:
    if config.initial_failures < 1:
        raise ValueError("initial_failures must be >= 1")
    if config.max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    rng = random.Random(config.seed)

    # Normalize node IDs to strings to avoid subtle int/str mismatches.
    g = nx.relabel_nodes(graph, lambda n: str(n), copy=True)

    base_load = node_metrics.betweenness
    capacity: dict[str, float] = {
        n: (1.0 + config.alpha) * float(base_load.get(n, 0.0)) for n in g.nodes
    }

    failed: set[str] = set()
    failed_step: dict[str, int] = {}

    initial = _select_initial_failures(g, base_load=base_load, config=config, rng=rng)
    for n in initial:
        failed.add(n)
        failed_step[n] = 0

    step_rows: list[dict[str, object]] = []

    current_graph = g.copy()
    current_graph.remove_nodes_from(initial)

    for step in range(1, config.max_steps + 1):
        # recompute load on remaining graph
        load = (
            nx.betweenness_centrality(current_graph, normalized=True)
            if current_graph.number_of_nodes()
            else {}
        )

        newly_failed: list[str] = []
        for n in current_graph.nodes:
            # n is already a str because of relabeling
            if float(load.get(n, 0.0)) > float(capacity.get(n, 0.0)):
                newly_failed.append(n)

        step_rows.append(
            {
                "step": step,
                "remaining_nodes": int(current_graph.number_of_nodes()),
                "new_failures": int(len(newly_failed)),
            }
        )

        if not newly_failed:
            break

        for n in newly_failed:
            failed.add(n)
            failed_step[n] = step
        current_graph.remove_nodes_from(newly_failed)

    nodes_rows: list[dict[str, object]] = []
    for n in g.nodes:
        n_str = n
        nodes_rows.append(
            {
                "node_id": n_str,
                "load0": float(base_load.get(n_str, 0.0)),
                "capacity": float(capacity.get(n_str, 0.0)),
                "failed": n_str in failed,
                "failed_step": failed_step.get(n_str),
            }
        )

    nodes_df = pl.from_dicts(nodes_rows)
    if step_rows:
        steps_df = pl.from_dicts(step_rows)
    else:
        steps_df = pl.DataFrame({"step": [], "remaining_nodes": [], "new_failures": []})

    return CascadeResult(nodes=nodes_df, steps=steps_df, config=config)


def _select_initial_failures(
    graph: nx.Graph,
    *,
    base_load: dict[str, float],
    config: CascadeConfig,
    rng: random.Random,
) -> list[str]:
    nodes = [str(n) for n in graph.nodes]
    k = min(config.initial_failures, len(nodes))

    if config.mode == FailureMode.random:
        rng.shuffle(nodes)
        return nodes[:k]

    # targeted: highest base load (betweenness)
    nodes_sorted = sorted(nodes, key=lambda n: float(base_load.get(n, 0.0)), reverse=True)
    return nodes_sorted[:k]
