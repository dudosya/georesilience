from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from georesilience.io.parquet import read_parquet
from georesilience.ml.deps import require_ml_deps


@dataclass(frozen=True)
class BuiltDataset:
    data: Any
    node_ids: list[str]
    feature_names: list[str]


def build_pyg_dataset(
    *,
    dataset_dir: Path,
    run_dir: Path,
    label_from_simulation: bool = True,
) -> BuiltDataset:
    """Build a Torch Geometric Data object.

    Features:
    - topological metrics from node_metrics.parquet
    - spatial coords from nodes.parquet (x, y)

    Labels:
    - from simulation_nodes.parquet (failed -> 0/1) if label_from_simulation
    """

    require_ml_deps()
    import torch
    from torch_geometric.data import Data

    nodes = read_parquet(dataset_dir / "nodes.parquet")
    edges = read_parquet(dataset_dir / "edges.parquet")

    node_metrics = read_parquet(run_dir / "node_metrics.parquet")

    # Join metrics with spatial coords.
    df = node_metrics.join(nodes.select(["node_id", "x", "y"]), on="node_id", how="left")

    feature_cols = [c for c in df.columns if c not in {"node_id"}]
    # Ensure stable ordering.
    preferred = [
        "degree",
        "closeness",
        "betweenness",
        "component_id",
        "component_size",
        "eigenvector",
        "x",
        "y",
    ]
    feature_cols = [c for c in preferred if c in feature_cols] + [
        c
        for c in feature_cols
        if c
        not in {
            "degree",
            "closeness",
            "betweenness",
            "component_id",
            "component_size",
            "eigenvector",
            "x",
            "y",
        }
    ]

    # Fill nulls with 0 for numeric features.
    df = df.with_columns([pl.col(c).fill_null(0) for c in feature_cols])

    node_ids = df["node_id"].to_list()
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    x_np = df.select(feature_cols).to_numpy()
    x = torch.tensor(x_np, dtype=torch.float32)

    # Store normalization stats; model will standardize internally.
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    std = torch.where(std > 0, std, torch.ones_like(std))

    # Build undirected edge_index
    u = edges["u"].cast(pl.Utf8).to_list()
    v = edges["v"].cast(pl.Utf8).to_list()
    edge_pairs: list[tuple[int, int]] = []
    for uu, vv in zip(u, v, strict=False):
        if uu in node_index and vv in node_index:
            ui = node_index[uu]
            vi = node_index[vv]
            edge_pairs.append((ui, vi))
            edge_pairs.append((vi, ui))

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = None
    if label_from_simulation:
        sim_nodes = read_parquet(run_dir / "simulation_nodes.parquet")
        sim = sim_nodes.select(["node_id", "failed"]).with_columns(
            pl.col("failed").cast(pl.Int8)
        )
        sim = sim.join(df.select(["node_id"]), on="node_id", how="right")
        sim = sim.with_columns(pl.col("failed").fill_null(0))
        y_np = sim["failed"].to_numpy()
        y = torch.tensor(y_np, dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = y

    # store normalization buffers on data for convenience
    data.x_mean = mean
    data.x_std = std

    return BuiltDataset(data=data, node_ids=node_ids, feature_names=feature_cols)
